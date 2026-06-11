"""
Improved trainer with:
  1. Extended feature set (advanced.py features)
  2. Class-balanced models to fix near-zero draw recall
  3. Stacking ensemble (meta-learner on top of base model probabilities)
  4. Time-series cross-validation for hyperparameter tuning
  5. Calibrated probabilities (Platt scaling / isotonic regression)
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

from src.features.builder import FEATURE_COLS
from src.features.advanced import ADVANCED_FEATURE_COLS
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_FEATURE_COLS = FEATURE_COLS + ADVANCED_FEATURE_COLS


def time_split(feat: pd.DataFrame, cutoff_year: int = 2018):
    train = feat[feat["date"].dt.year < cutoff_year].copy()
    test  = feat[feat["date"].dt.year >= cutoff_year].copy()
    logger.info(f"Train: {len(train):,} | Test: {len(test):,} (cutoff={cutoff_year})")
    return train, test


def _prep(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df["target"].values
    return X, y, available


# ---------------------------------------------------------------------------
# Class weight helper: upweights draw class to fix near-zero draw recall
# ---------------------------------------------------------------------------
def _draw_aware_class_weight(y, draw_multiplier: float = 2.5):
    """
    Return class weights that upweight draws by `draw_multiplier`.
    Without this, all tree-based and linear models collapse draws to ~0%
    recall because draws are 24% of data but 'noisy'.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    weight_dict[1] *= draw_multiplier   # class 1 = draw
    return weight_dict


def _build_base_models(y_train):
    cw = _draw_aware_class_weight(y_train)

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, C=0.8,
                class_weight=cw,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            class_weight=cw, n_jobs=-1, random_state=42,
        ),
    }
    if _HAS_XGB:
        # XGBoost doesn't take class_weight directly — use scale_pos_weight per class
        # or sample_weight during fit.  We pass it via sample_weight in train_all.
        models["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, gamma=0.1,
            min_child_weight=5,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
            verbosity=0,
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            min_child_samples=10,
            class_weight=cw,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if _HAS_CAT:
        models["catboost"] = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.03,
            class_weights=[cw.get(0, 1), cw.get(1, 1), cw.get(2, 1)],
            random_seed=42, verbose=0,
        )
    return models


def _sample_weights(y, draw_multiplier: float = 2.5):
    """Per-sample weights for XGBoost/other models that don't support class_weight."""
    w = np.ones(len(y))
    balanced = len(y) / (3 * np.bincount(y)[y])
    w *= balanced
    w[y == 1] *= draw_multiplier
    return w


def _calibrate(model, X_cal, y_cal):
    """Platt-scale probabilities on a calibration slice."""
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def _evaluate(name, model, X_test, y_test, feature_cols):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    ll   = log_loss(y_test, y_prob)
    brier = np.mean([
        brier_score_loss((y_test == cls).astype(int), y_prob[:, i])
        for i, cls in enumerate(model.classes_)
    ])

    cm = confusion_matrix(y_test, y_pred)
    draw_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    logger.info(
        f"[{name}] Acc={acc:.4f}  LogLoss={ll:.4f}  Brier={brier:.4f}  "
        f"DrawRecall={draw_recall:.3f}"
    )
    return {
        "model_name": name,
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "draw_recall": draw_recall,
        "n_features": len(feature_cols),
        "confusion_matrix": cm.tolist(),
    }


def build_stacking_ensemble(base_models: dict, X_train, y_train):
    """
    Stack the base classifiers with a Logistic Regression meta-learner.
    The meta-learner sees the probability outputs of all base models and learns
    the optimal blend — especially useful for draw prediction where models
    disagree.
    """
    logger.info("Building stacking ensemble ...")
    estimators = [(name, model) for name, model in base_models.items()]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, C=1.0),
        cv=5,
        stack_method="predict_proba",
        passthrough=True,   # also pass original features to meta-learner
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    logger.info("Stacking ensemble trained.")
    return stack


def train_all_v2(feat: pd.DataFrame, cutoff_year: int = 2018,
                 use_advanced: bool = True, build_ensemble: bool = True):
    """
    Full v2 training pipeline.

    Returns
    -------
    trained  : dict of name → fitted model
    results  : DataFrame with evaluation metrics
    feat_cols: list of feature columns actually used
    """
    train, test = time_split(feat, cutoff_year)

    # Use extended features if available
    feature_cols = ALL_FEATURE_COLS if use_advanced else FEATURE_COLS
    X_train, y_train, feature_cols = _prep(train, feature_cols)
    X_test,  y_test,  _            = _prep(test,  feature_cols)

    # Reserve last 10% of train for calibration
    cal_split = int(len(X_train) * 0.9)
    X_fit,  y_fit  = X_train[:cal_split], y_train[:cal_split]
    X_cal,  y_cal  = X_train[cal_split:], y_train[cal_split:]

    sw_fit = _sample_weights(y_fit)

    trained, rows = {}, []
    base_models = _build_base_models(y_fit)

    for name, model in base_models.items():
        logger.info(f"Training {name} ...")
        if name in ("xgboost",):
            model.fit(X_fit, y_fit, sample_weight=sw_fit)
        else:
            model.fit(X_fit, y_fit)

        # Calibrate
        model_cal = _calibrate(model, X_cal, y_cal)
        metrics = _evaluate(f"{name}_cal", model_cal, X_test, y_test, feature_cols)
        trained[name] = model_cal
        rows.append(metrics)

    # Stacking ensemble
    if build_ensemble and len(base_models) >= 2:
        # Re-fit uncalibrated base models on full train for the stack
        raw_bases = _build_base_models(y_train)
        for name, m in raw_bases.items():
            if name == "xgboost":
                m.fit(X_train, y_train, sample_weight=_sample_weights(y_train))
            else:
                m.fit(X_train, y_train)

        ensemble = build_stacking_ensemble(raw_bases, X_train, y_train)
        metrics = _evaluate("ensemble", ensemble, X_test, y_test, feature_cols)
        trained["ensemble"] = ensemble
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("model_name")

    logger.info("\n--- V2 Model Comparison ---")
    logger.info(
        results[["accuracy", "log_loss", "brier_score", "draw_recall", "n_features"]]
        .to_string()
    )
    return trained, results, feature_cols


def save_models_v2(trained: dict, feature_cols: list) -> None:
    for name, model in trained.items():
        path = MODELS_DIR / f"{name}_v2.joblib"
        joblib.dump(model, path)
        logger.info(f"Saved {name}_v2 -> {path}")

    # Save feature column list so predictor knows which features to use
    import json
    feat_path = MODELS_DIR / "feature_cols_v2.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"Feature list saved -> {feat_path}")

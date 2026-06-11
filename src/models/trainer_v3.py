"""
V3 trainer — the definitive improved pipeline.

Key insights from V1/V2 analysis
----------------------------------
1. Log-loss 0.880 (V1 LogReg) is already competitive — professional bookmakers
   hit ~0.85–0.90 on this feature set.  The theoretical ceiling with historical
   stats only is ~63-65% accuracy; player-level data would push it higher.

2. Draw recall is near zero in all argmax classifiers because draws are the
   "middle" class — the model probability is rarely the highest.  The fix is
   THRESHOLD CALIBRATION, not class weighting (which hurts log-loss).

3. Stacking WITHOUT isotonic calibration is the best ensemble strategy.

4. V2 features (elo_closeness, draw_tendency, days_rest, elo_momentum) add
   signal specifically for draw prediction — they need to be combined with
   proper threshold tuning to show up in evaluation.

Changes vs V1/V2
-----------------
- Remove isotonic calibration (overfits on small holdout)
- Use Platt scaling (sigmoid, data-efficient) via CalibratedClassifierCV(cv=5)
- Add draw-probability threshold (tuned on val set, default 0.270)
- Clean stacking with LogReg meta-learner + no calibration
- Per-class reporting: home/draw/away precision, recall, F1
"""
import joblib, json, warnings
import numpy as np
import pandas as pd

# Suppress sklearn feature-name consistency warnings raised during CV calibration
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier; _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier; _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
try:
    from catboost import CatBoostClassifier; _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

from src.features.builder import FEATURE_COLS
from src.features.advanced import ADVANCED_FEATURE_COLS
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_FEATURE_COLS = FEATURE_COLS + ADVANCED_FEATURE_COLS


def time_split(feat, cutoff_year=2018):
    train = feat[feat["date"].dt.year < cutoff_year].copy()
    test  = feat[feat["date"].dt.year >= cutoff_year].copy()
    logger.info(f"Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def _prep(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS
    avail = [c for c in feature_cols if c in df.columns]
    return df[avail].fillna(0).values, df["target"].values, avail


def _tune_draw_threshold(y_val, y_prob, step=0.01):
    """
    Find the draw probability threshold that maximises overall accuracy on
    validation data.  With threshold t: predict draw if P(draw) >= t, else
    pick the higher of P(home) vs P(away).
    """
    best_thresh, best_acc = 0.33, 0.0
    for t in np.arange(0.18, 0.40, step):
        pred = np.where(
            y_prob[:, 1] >= t, 1,
            np.where(y_prob[:, 0] > y_prob[:, 2], 0, 2)
        )
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc


def _build_models():
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.8)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            n_jobs=-1, random_state=42,
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, gamma=0.1,
            min_child_weight=5, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if _HAS_CAT:
        models["catboost"] = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.03,
            random_seed=42, verbose=0,
        )
    return models


def _calibrate_platt(model, X_train, y_train, feature_cols=None):
    """Platt (sigmoid) calibration with 5-fold CV — much better than prefit isotonic."""
    import warnings
    # LightGBM expects a DataFrame; wrap if feature_cols provided
    X_fit = pd.DataFrame(X_train, columns=feature_cols) if feature_cols else X_train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = CalibratedClassifierCV(model, method="sigmoid", cv=5, n_jobs=-1)
        cal.fit(X_fit, y_train)
    return cal


def _evaluate_full(name, model, X_test, y_test, draw_thresh):
    y_prob = model.predict_proba(X_test)

    # Argmax prediction (standard)
    y_pred_argmax = model.predict(X_test)

    # Threshold-based prediction
    y_pred_thresh = np.where(
        y_prob[:, 1] >= draw_thresh, 1,
        np.where(y_prob[:, 0] > y_prob[:, 2], 0, 2)
    )

    ll    = log_loss(y_test, y_prob)
    brier = np.mean([
        brier_score_loss((y_test == c).astype(int), y_prob[:, i])
        for i, c in enumerate(model.classes_)
    ])

    acc_argmax = accuracy_score(y_test, y_pred_argmax)
    acc_thresh = accuracy_score(y_test, y_pred_thresh)

    cm = confusion_matrix(y_test, y_pred_thresh)
    draw_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    logger.info(
        f"[{name}] Acc(argmax)={acc_argmax:.4f}  Acc(thresh={draw_thresh:.2f})={acc_thresh:.4f}  "
        f"LogLoss={ll:.4f}  Brier={brier:.4f}  DrawRecall={draw_recall:.3f}"
    )
    return {
        "model_name": name,
        "accuracy_argmax": acc_argmax,
        "accuracy_with_draw_threshold": acc_thresh,
        "log_loss": ll,
        "brier_score": brier,
        "draw_recall": draw_recall,
        "draw_threshold": draw_thresh,
        "confusion_matrix": cm.tolist(),
    }


def train_all_v3(feat, cutoff_year=2018, build_ensemble=True):
    """
    Full V3 training:
    1. Fit base models with Platt calibration (5-fold CV)
    2. Tune draw threshold on last 20% of training data
    3. Build stacking ensemble
    4. Report both argmax accuracy and threshold-tuned accuracy
    """
    train, test = time_split(feat, cutoff_year)
    X_train, y_train, feature_cols = _prep(train)
    X_test,  y_test,  _            = _prep(test, feature_cols)

    # Validation slice for threshold tuning (last 20% of train)
    val_n = int(len(X_train) * 0.2)
    X_fit, y_fit = X_train[:-val_n], y_train[:-val_n]
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]

    trained, rows = {}, []
    base_models = _build_models()

    for name, model in base_models.items():
        logger.info(f"Training {name} (Platt calibration, 5-fold) ...")
        cal = _calibrate_platt(model, X_train, y_train, feature_cols)

        # Tune draw threshold on val set
        y_prob_val = cal.predict_proba(X_val)
        draw_thresh, _ = _tune_draw_threshold(y_val, y_prob_val)

        metrics = _evaluate_full(name, cal, X_test, y_test, draw_thresh)
        trained[name] = cal
        trained[f"{name}_draw_thresh"] = draw_thresh
        rows.append(metrics)

    # Stacking ensemble (use integer cv — TimeSeriesSplit breaks StackingClassifier)
    if build_ensemble and len(base_models) >= 2:
        logger.info("Building stacking ensemble ...")
        estimators = [(n, _build_models()[n]) for n in base_models]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, C=1.0),
            cv=5,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=1,   # avoid multiprocess pickling issues on Windows
        )
        stack.fit(X_train, y_train)
        y_prob_val_stack = stack.predict_proba(X_val)
        draw_thresh_stack, _ = _tune_draw_threshold(y_val, y_prob_val_stack)
        metrics = _evaluate_full("ensemble", stack, X_test, y_test, draw_thresh_stack)
        trained["ensemble"] = stack
        trained["ensemble_draw_thresh"] = draw_thresh_stack
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("model_name")
    logger.info("\n--- V3 Final Results ---")
    logger.info(
        results[["accuracy_argmax", "accuracy_with_draw_threshold",
                 "log_loss", "brier_score", "draw_recall"]].to_string()
    )
    return trained, results, feature_cols


def save_models_v3(trained, feature_cols):
    thresholds = {}
    for name, obj in trained.items():
        if name.endswith("_draw_thresh"):
            thresholds[name.replace("_draw_thresh", "")] = obj
            continue
        path = MODELS_DIR / f"{name}_v3.joblib"
        joblib.dump(obj, path)
        logger.info(f"Saved {name}_v3 -> {path}")

    meta = {"feature_cols": feature_cols, "draw_thresholds": thresholds}
    meta_path = MODELS_DIR / "model_meta_v3.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model metadata saved -> {meta_path}")

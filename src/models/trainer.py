"""
Train and evaluate multiple classifiers for match outcome prediction.

Outcome classes:
  2 = home win,  1 = draw,  0 = away win

Time-based split is used to avoid data leakage:
  - Train on matches before the cutoff date
  - Test on matches from the cutoff date onward
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

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
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def time_split(feat: pd.DataFrame, cutoff_year: int = 2018):
    """Split by year to prevent future data leaking into training."""
    train = feat[feat["date"].dt.year < cutoff_year].copy()
    test = feat[feat["date"].dt.year >= cutoff_year].copy()
    logger.info(f"Train: {len(train):,} | Test: {len(test):,} (cutoff={cutoff_year})")
    return train, test


def _prep(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0).values
    y = df["target"].values
    return X, y


def _build_models() -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            n_jobs=-1, random_state=42,
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if _HAS_CAT:
        models["catboost"] = CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
    return models


def _evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)

    # Brier score: average over classes
    brier = np.mean([
        brier_score_loss((y_test == cls).astype(int), y_prob[:, i])
        for i, cls in enumerate(model.classes_)
    ])

    logger.info(
        f"[{name}] Accuracy={acc:.4f}  LogLoss={ll:.4f}  Brier={brier:.4f}"
    )
    return {
        "model_name": name,
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["away_win", "draw", "home_win"], output_dict=True
        ),
    }


def train_all(feat: pd.DataFrame, cutoff_year: int = 2018) -> tuple[dict, pd.DataFrame]:
    """
    Train all available models, evaluate on hold-out test set.

    Returns
    -------
    trained_models : dict of name → fitted model
    results        : DataFrame with evaluation metrics per model
    """
    train, test = time_split(feat, cutoff_year)
    X_train, y_train = _prep(train)
    X_test, y_test = _prep(test)

    trained, rows = {}, []
    for name, model in _build_models().items():
        logger.info(f"Training {name} …")
        model.fit(X_train, y_train)
        metrics = _evaluate(name, model, X_test, y_test)
        trained[name] = model
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("model_name")
    return trained, results


def save_models(trained: dict) -> None:
    for name, model in trained.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Saved {name} -> {path}")


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


def load_best_model(results: pd.DataFrame):
    """Load the model with lowest log loss."""
    best_name = results["log_loss"].idxmin()
    logger.info(f"Best model by log-loss: {best_name}")
    return load_model(best_name), best_name

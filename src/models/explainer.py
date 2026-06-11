"""
Model explainability: feature importance and SHAP values.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.features.builder import FEATURE_COLS
from src.utils.config import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_importance(model, model_name: str) -> pd.Series:
    """
    Extract feature importances from tree-based models; fallback to
    abs(coef) for linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline (e.g. LogisticRegression wrapped in StandardScaler)
        clf = model.named_steps.get("clf", None)
        if clf and hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            return pd.Series(dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        logger.warning(f"Cannot extract importances from {model_name}")
        return pd.Series(dtype=float)

    return pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, model_name: str, top_n: int = 20) -> Path:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = REPORTS_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot saved -> {path}")
    return path


def compute_shap_values(model, X: np.ndarray, model_name: str):
    """
    Compute SHAP values if shap is installed and the model supports it.
    Returns (shap_values, explainer) or (None, None).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis.")
        return None, None

    try:
        if "xgboost" in model_name or "lightgbm" in model_name or "random_forest" in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X, feature_perturbation="correlation_dependent")

        shap_values = explainer.shap_values(X)
        logger.info(f"SHAP values computed for {model_name}")
        return shap_values, explainer
    except Exception as exc:
        logger.warning(f"SHAP failed for {model_name}: {exc}")
        return None, None


def plot_shap_summary(shap_values, X: np.ndarray, model_name: str) -> Path | None:
    try:
        import shap
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")

        if isinstance(shap_values, list):
            sv = shap_values[2]  # class 2 = home win
        else:
            sv = shap_values

        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv, X,
            feature_names=FEATURE_COLS,
            plot_type="bar",
            show=False,
        )
        path = REPORTS_DIR / f"shap_summary_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved -> {path}")
        return path
    except Exception as exc:
        logger.warning(f"SHAP plot failed: {exc}")
        return None

"""
V3 training — definitive pipeline with threshold-tuned draw prediction + stacking.

Usage
-----
  python train_v3.py             # full run (uses cached features_v2.csv)
  python train_v3.py --no-stack  # skip stacking ensemble (faster, ~2 min)
  python train_v3.py --audit     # full performance audit vs baselines
"""
import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import DATA_PROCESSED, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="reports/pipeline_v3.log")


def audit_model(model_name, model, X_test, y_test, draw_thresh=None):
    """Print a full breakdown of model performance."""
    from sklearn.metrics import confusion_matrix, classification_report
    import json

    y_prob = model.predict_proba(X_test)

    if draw_thresh is None:
        draw_thresh = 0.27  # sensible default

    y_pred = np.where(
        y_prob[:, 1] >= draw_thresh, 1,
        np.where(y_prob[:, 0] > y_prob[:, 2], 0, 2)
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
    )
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}  (draw_thresh={draw_thresh:.2f})")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Away  Pred Draw  Pred Home")
    print(f"Actual Away     {cm[0,0]:6d}     {cm[0,1]:6d}     {cm[0,2]:6d}")
    print(f"Actual Draw     {cm[1,0]:6d}     {cm[1,1]:6d}     {cm[1,2]:6d}")
    print(f"Actual Home     {cm[2,0]:6d}     {cm[2,1]:6d}     {cm[2,2]:6d}")
    print(f"\nClassification Report:\n{report}")


def run_v3(no_stack=False, audit=False):
    from src.models.trainer_v3 import train_all_v3, save_models_v3
    from src.features.builder import FEATURE_COLS
    from src.features.advanced import ADVANCED_FEATURE_COLS

    # Load advanced features (from train_v2.py run)
    v2_path = DATA_PROCESSED / "features_v2.csv"
    v1_path = DATA_PROCESSED / "features.csv"

    if v2_path.exists():
        logger.info("Loading advanced features (v2) ...")
        feat = pd.read_csv(v2_path, parse_dates=["date"])
    elif v1_path.exists():
        logger.info("Advanced features not found — using base features. Run train_v2.py --no-h2h first for best results.")
        feat = pd.read_csv(v1_path, parse_dates=["date"])
    else:
        logger.error("No feature file found. Run python main.py first.")
        sys.exit(1)

    logger.info(f"Feature set: {feat.shape[1]-4} features, {len(feat):,} matches")

    trained, results, feature_cols = train_all_v3(feat, cutoff_year=2018,
                                                   build_ensemble=not no_stack)
    save_models_v3(trained, feature_cols)

    # Save results
    results_path = Path("reports") / "model_results_v3.csv"
    results.to_csv(results_path)

    # Compare all versions
    logger.info("\n" + "="*70)
    logger.info("COMPARISON: V1 baseline  vs  V3 best")
    logger.info("="*70)

    v1_path_r = Path("reports") / "model_results.csv"
    if v1_path_r.exists():
        v1 = pd.read_csv(v1_path_r, index_col=0)
        best_v1_ll  = v1["log_loss"].min()
        best_v1_acc = v1["accuracy"].max()
    else:
        best_v1_ll = best_v1_acc = None

    best_v3_ll  = results["log_loss"].min()
    best_v3_acc = results["accuracy_with_draw_threshold"].max()
    best_v3_dr  = results["draw_recall"].max()

    logger.info(f"V1  — Accuracy: {best_v1_acc:.4f}  LogLoss: {best_v1_ll:.4f}  DrawRecall: ~0.005")
    logger.info(f"V3  — Accuracy: {best_v3_acc:.4f}  LogLoss: {best_v3_ll:.4f}  DrawRecall: {best_v3_dr:.3f}")

    if best_v1_ll:
        ll_improvement = (best_v1_ll - best_v3_ll) / best_v1_ll * 100
        acc_improvement = (best_v3_acc - best_v1_acc) * 100
        logger.info(f"\nLog-Loss change:  {ll_improvement:+.2f}%")
        logger.info(f"Accuracy change:  {acc_improvement:+.2f} pp")

    logger.info("""
Performance context:
  Random baseline         33.3%
  Always-predict-home     47.2%
  V1 best (logistic)      59.4%   draw recall ~0%
  V3 best (ensemble)     see above  draw recall ~25-35%
  Professional bookmakers ~60-65%  (with betting odds + injuries)
  Theoretical ceiling     ~65%     (historical stats only)

  To go above 65%, you would need:
    - Squad availability / injury data
    - Market odds (already encode expert probability)
    - Weather / altitude conditions
    - Individual player Elo ratings
""")

    if audit:
        test_feat = feat[feat["date"].dt.year >= 2018]
        ALL = FEATURE_COLS + ADVANCED_FEATURE_COLS
        avail = [c for c in ALL if c in feat.columns]
        X_test = test_feat[avail].fillna(0).values
        y_test = test_feat["target"].values

        meta_path = MODELS_DIR / "model_meta_v3.json"
        thresholds = {}
        if meta_path.exists():
            with open(meta_path) as f:
                thresholds = json.load(f).get("draw_thresholds", {})

        best_model_name = results["log_loss"].idxmin()
        import joblib
        best_model = joblib.load(MODELS_DIR / f"{best_model_name}_v3.joblib")
        thresh = thresholds.get(best_model_name, 0.27)
        audit_model(best_model_name, best_model, X_test, y_test, thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stack", action="store_true")
    parser.add_argument("--audit",    action="store_true")
    args = parser.parse_args()
    run_v3(no_stack=args.no_stack, audit=args.audit)

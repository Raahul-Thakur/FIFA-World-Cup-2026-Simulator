"""
V2 training pipeline — runs advanced feature engineering + improved models.

Usage
-----
  python train_v2.py                  # full advanced pipeline
  python train_v2.py --no-h2h         # skip slow H2H computation
  python train_v2.py --no-ensemble    # skip stacking (faster)
  python train_v2.py --compare        # compare v1 vs v2 side-by-side
"""
import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import DATA_PROCESSED, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="reports/pipeline_v2.log")

ADVANCED_FEATURES_PATH = DATA_PROCESSED / "features_v2.csv"


def build_advanced_features(skip_h2h: bool = False) -> pd.DataFrame:
    logger.info("Loading base features ...")
    feat_path = DATA_PROCESSED / "features.csv"
    if not feat_path.exists():
        logger.error("Run `python main.py --skip-features` first to generate features.csv")
        sys.exit(1)

    feat = pd.read_csv(feat_path, parse_dates=["date"])

    # Load the full clean match data (needed for H2H and raw scores)
    clean_path = DATA_PROCESSED / "matches_clean.csv"
    df_clean = pd.read_csv(clean_path, parse_dates=["date"])

    # ── Elo features (need elo_before columns from elo.py) ────────────────
    logger.info("Adding Elo-derived features ...")
    from src.features.advanced import compute_elo_features
    feat = compute_elo_features(feat)

    # ── Days rest ─────────────────────────────────────────────────────────
    logger.info("Adding days-rest features ...")
    from src.features.advanced import compute_days_rest
    feat = compute_days_rest(feat)

    # ── Draw tendency (vectorised) ─────────────────────────────────────────
    logger.info("Adding draw-tendency features ...")
    from src.features.advanced import compute_draw_tendency
    # Merge raw draw info from clean dataset
    draw_info = df_clean[["date", "home_team", "away_team",
                           "home_score", "away_score"]].copy()
    # draw_tendency needs the raw scores — compute on feat after merging scores
    feat_with_scores = feat.merge(
        draw_info.rename(columns={
            "home_score": "_hs", "away_score": "_as"
        }),
        on=["date", "home_team", "away_team"], how="left"
    )
    # Temporarily add score columns
    feat_with_scores["home_score"] = feat_with_scores["_hs"]
    feat_with_scores["away_score"] = feat_with_scores["_as"]
    feat_with_scores = compute_draw_tendency(feat_with_scores, window=20)
    for col in ["home_draw_tendency", "away_draw_tendency", "combined_draw_tendency"]:
        feat[col] = feat_with_scores[col].values

    # ── Head-to-head (row-by-row, but cheap) ─────────────────────────────
    if not skip_h2h:
        logger.info("Adding head-to-head features (this may take ~2 min) ...")
        from src.features.advanced import compute_h2h_features
        # Pass clean df with scores for H2H computation
        feat_with_scores2 = feat.copy()
        feat_with_scores2["home_score"] = feat_with_scores["home_score"].values
        feat_with_scores2["away_score"] = feat_with_scores["away_score"].values
        feat_with_scores2 = compute_h2h_features(feat_with_scores2)
        h2h_cols = ["h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
                    "h2h_home_gd", "h2h_n"]
        for col in h2h_cols:
            feat[col] = feat_with_scores2[col].values
    else:
        logger.info("Skipping H2H (--no-h2h flag set)")
        for col in ["h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
                    "h2h_home_gd", "h2h_n"]:
            feat[col] = 0.0

    feat.to_csv(ADVANCED_FEATURES_PATH, index=False)
    logger.info(f"Advanced features saved -> {ADVANCED_FEATURES_PATH}")
    return feat


def run_v2(skip_h2h: bool = False, no_ensemble: bool = False, compare: bool = False):
    logger.info("=" * 60)
    logger.info("V2 PIPELINE: Advanced Features + Improved Models")
    logger.info("=" * 60)

    if ADVANCED_FEATURES_PATH.exists():
        logger.info("Loading cached advanced features ...")
        feat = pd.read_csv(ADVANCED_FEATURES_PATH, parse_dates=["date"])
    else:
        feat = build_advanced_features(skip_h2h=skip_h2h)

    logger.info("=" * 60)
    logger.info("Training V2 models ...")
    logger.info("=" * 60)

    from src.models.trainer_v2 import train_all_v2, save_models_v2
    trained, results, feature_cols = train_all_v2(
        feat,
        cutoff_year=2018,
        use_advanced=True,
        build_ensemble=not no_ensemble,
    )
    save_models_v2(trained, feature_cols)

    # Save results
    results_path = Path("reports") / "model_results_v2.csv"
    results_path.parent.mkdir(exist_ok=True)
    results[["accuracy", "log_loss", "brier_score", "draw_recall"]].to_csv(results_path)
    logger.info(f"Results saved -> {results_path}")

    if compare:
        v1_path = Path("reports") / "model_results.csv"
        if v1_path.exists():
            v1 = pd.read_csv(v1_path, index_col=0)
            logger.info("\n--- V1 Results ---")
            logger.info(v1.to_string())
            logger.info("\n--- V2 Results ---")
            logger.info(results[["accuracy", "log_loss", "brier_score", "draw_recall"]].to_string())

            # Print improvement
            best_v1_ll = v1["log_loss"].min()
            best_v2_ll = results["log_loss"].min()
            best_v1_acc = v1["accuracy"].max()
            best_v2_acc = results["accuracy"].max()
            logger.info(
                f"\nImprovement: LogLoss {best_v1_ll:.4f} -> {best_v2_ll:.4f} "
                f"({(best_v1_ll - best_v2_ll)*100:.2f}% better)"
            )
            logger.info(
                f"Improvement: Accuracy {best_v1_acc:.4f} -> {best_v2_acc:.4f} "
                f"({(best_v2_acc - best_v1_acc)*100:.2f}pp better)"
            )

    logger.info("=" * 60)
    logger.info("V2 pipeline complete. Dashboard will auto-load best v2 model.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Training Pipeline")
    parser.add_argument("--no-h2h", action="store_true", help="Skip H2H features (faster)")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip stacking ensemble")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2 results")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Recompute advanced features even if cached")
    args = parser.parse_args()

    if args.rebuild_features and ADVANCED_FEATURES_PATH.exists():
        ADVANCED_FEATURES_PATH.unlink()
        logger.info("Cleared cached advanced features.")

    run_v2(
        skip_h2h=args.no_h2h,
        no_ensemble=args.no_ensemble,
        compare=args.compare,
    )

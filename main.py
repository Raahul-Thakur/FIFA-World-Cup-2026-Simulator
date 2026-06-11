"""
FIFA World Cup Predictor — end-to-end pipeline runner.

Usage
-----
  python main.py                  # full pipeline (download → feature → train → simulate)
  python main.py --skip-features  # re-use cached feature file
  python main.py --simulate-only  # skip training, just run Monte Carlo
  python main.py --fast           # small simulation run for quick testing
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

from src.utils.config import DATA_PROCESSED, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="reports/pipeline.log")


def run_pipeline(skip_features: bool = False, simulate_only: bool = False, fast: bool = False):
    # ------------------------------------------------------------------ #
    # Phase 1 — Data ingestion                                            #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Ingestion")
    logger.info("=" * 60)

    from src.data.downloader import load_or_generate_results
    from src.data.preprocessor import load_and_clean, filter_modern_era, save_processed

    df_raw = load_or_generate_results()
    df_clean = load_and_clean(df_raw)
    df_modern = filter_modern_era(df_clean, from_year=1993)
    save_processed(df_modern, "matches_clean.csv")

    if simulate_only:
        logger.info("--simulate-only: skipping feature engineering and model training.")
        _run_simulation(fast)
        return

    # ------------------------------------------------------------------ #
    # Phase 2 — Feature Engineering                                       #
    # ------------------------------------------------------------------ #
    feature_path = DATA_PROCESSED / "features.csv"
    if skip_features and feature_path.exists():
        logger.info("=" * 60)
        logger.info("PHASE 2: Loading cached features")
        logger.info("=" * 60)
        feat = pd.read_csv(feature_path, parse_dates=["date"])
    else:
        logger.info("=" * 60)
        logger.info("PHASE 2: Feature Engineering")
        logger.info("=" * 60)

        from src.features.elo import compute_elo_ratings
        from src.features.form import compute_form_features
        from src.features.builder import build_feature_matrix, save_features

        df_elo, current_elos = compute_elo_ratings(df_modern)

        # Cache current Elos for the simulator
        elo_series = pd.Series(current_elos, name="elo").rename_axis("team")
        elo_series.to_csv(DATA_PROCESSED / "current_elos.csv")
        logger.info(f"Current Elos saved for {len(current_elos)} teams.")

        df_form = compute_form_features(df_elo, windows=[5, 10])
        feat = build_feature_matrix(df_form)
        save_features(feat, "features.csv")

    # ------------------------------------------------------------------ #
    # Phase 3 — Model Training                                            #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PHASE 3: Model Training")
    logger.info("=" * 60)

    from src.models.trainer import train_all, save_models
    from src.models.explainer import get_feature_importance, plot_feature_importance

    trained, results = train_all(feat, cutoff_year=2018)
    save_models(trained)

    logger.info("\n--- Model Comparison ---")
    logger.info(results[["accuracy", "log_loss", "brier_score"]].to_string())

    # Save results table
    results_path = Path("reports") / "model_results.csv"
    results_path.parent.mkdir(exist_ok=True)
    results[["accuracy", "log_loss", "brier_score"]].to_csv(results_path)

    # Feature importance plots
    for name, model in trained.items():
        imp = get_feature_importance(model, name)
        if not imp.empty:
            plot_feature_importance(imp, name)

    # ------------------------------------------------------------------ #
    # Phase 4 — Tournament Simulation                                     #
    # ------------------------------------------------------------------ #
    _run_simulation(fast)

    logger.info("=" * 60)
    logger.info("Pipeline complete.  Run `streamlit run app/dashboard.py` to view the dashboard.")
    logger.info("=" * 60)


def _run_simulation(fast: bool = False):
    logger.info("=" * 60)
    logger.info("PHASE 4: Tournament Simulation")
    logger.info("=" * 60)

    from src.simulation.simulator import run_simulation, GROUPS_2022
    from src.utils.config import DATA_PROCESSED, N_SIMULATIONS

    # Load current Elos if available
    elo_path = DATA_PROCESSED / "current_elos.csv"
    elo = {}
    if elo_path.exists():
        elo_df = pd.read_csv(elo_path, index_col="team")
        elo = elo_df["elo"].to_dict()
        logger.info(f"Loaded {len(elo)} current Elo ratings.")

    n_sims = 1000 if fast else N_SIMULATIONS
    sim_results = run_simulation(
        groups=GROUPS_2022,
        elo=elo or None,
        n_simulations=n_sims,
    )

    out_path = DATA_PROCESSED / "simulation_results.csv"
    sim_results.to_csv(out_path, index=False)
    logger.info(f"Simulation results saved -> {out_path}")

    logger.info("\n--- Top 10 Tournament Winner Probabilities ---")
    logger.info(
        sim_results[["team", "p_winner", "p_final", "p_sf", "p_qf", "p_r16"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIFA World Cup Predictor Pipeline")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering, use cached features.csv")
    parser.add_argument("--simulate-only", action="store_true",
                        help="Skip training, just run Monte Carlo simulation")
    parser.add_argument("--fast", action="store_true",
                        help="Use 1,000 simulations for quick testing")
    args = parser.parse_args()

    run_pipeline(
        skip_features=args.skip_features,
        simulate_only=args.simulate_only,
        fast=args.fast,
    )

"""
Rolling form features — fully vectorised implementation.

Each match row is expanded into two "team-centric" rows (home + away),
rolling stats are computed with groupby + rolling(shift=1), then the
results are pivoted back to home_* / away_* columns on the original index.
Runs in seconds on 30k matches.
"""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STAT_DEFAULTS = {"win": 0.4, "draw": 0.25, "loss": 0.35, "gf": 1.3, "ga": 1.0}


def compute_form_features(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10]

    df = df.sort_values("date").reset_index(drop=True)
    global_avg_gf = float(df["home_score"].mean())
    global_avg_ga = float(df["away_score"].mean())

    logger.info(f"Vectorised form computation for {len(df):,} matches ...")

    # ── 1. Build long (team-centric) table ───────────────────────────────
    home_side = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "date":  df["date"].values,
        "team":  df["home_team"].values,
        "gf":    df["home_score"].values,
        "ga":    df["away_score"].values,
        "win":   (df["home_score"] > df["away_score"]).astype(np.int8).values,
        "draw":  (df["home_score"] == df["away_score"]).astype(np.int8).values,
        "loss":  (df["home_score"] < df["away_score"]).astype(np.int8).values,
        "side":  "home",
    })
    away_side = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "date":  df["date"].values,
        "team":  df["away_team"].values,
        "gf":    df["away_score"].values,
        "ga":    df["home_score"].values,
        "win":   (df["away_score"] > df["home_score"]).astype(np.int8).values,
        "draw":  (df["home_score"] == df["away_score"]).astype(np.int8).values,
        "loss":  (df["away_score"] < df["home_score"]).astype(np.int8).values,
        "side":  "away",
    })

    long = pd.concat([home_side, away_side], ignore_index=True)
    long = long.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    stat_cols = ["win", "draw", "loss", "gf", "ga"]

    # ── 2. Rolling means (shift(1) = no leakage) ─────────────────────────
    grp = long.groupby("team", sort=False)
    for w in windows:
        for col in stat_cols:
            long[f"{col}_{w}"] = (
                grp[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                .fillna(_STAT_DEFAULTS[col])
            )
        long[f"gd_{w}"] = long[f"gf_{w}"] - long[f"ga_{w}"]

    long["atk_strength"] = (
        grp["gf"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        .fillna(global_avg_gf) / max(global_avg_gf, 1e-6)
    )
    long["def_strength"] = (
        grp["ga"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        .fillna(global_avg_ga) / max(global_avg_ga, 1e-6)
    )

    # ── 3. Pivot back to match level ──────────────────────────────────────
    feature_cols = (
        [f"{c}_{w}" for w in windows for c in stat_cols]
        + [f"gd_{w}" for w in windows]
        + ["atk_strength", "def_strength"]
    )

    for side in ("home", "away"):
        side_df = long[long["side"] == side].set_index("match_idx")[feature_cols]
        # Reindex to guarantee alignment with df
        side_df = side_df.reindex(range(len(df)))
        for w in windows:
            df[f"{side}_form_{w}_wins"]   = side_df[f"win_{w}"].values
            df[f"{side}_form_{w}_draws"]  = side_df[f"draw_{w}"].values
            df[f"{side}_form_{w}_losses"] = side_df[f"loss_{w}"].values
            df[f"{side}_form_{w}_gf"]     = side_df[f"gf_{w}"].values
            df[f"{side}_form_{w}_ga"]     = side_df[f"ga_{w}"].values
            df[f"{side}_form_{w}_gd"]     = side_df[f"gd_{w}"].values
        df[f"{side}_attack_strength"]  = side_df["atk_strength"].values
        df[f"{side}_defense_strength"] = side_df["def_strength"].values

    logger.info("Form features done.")
    return df

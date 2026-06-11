"""
Assemble the final feature matrix from Elo + form + contextual features.

The output is a single DataFrame where every row is one match and every column
is a feature ready for model training.  The target column is `target`:
  2 = home win,  1 = draw,  0 = away win

All features are *pre-match* (no future leakage).
"""
import pandas as pd
import numpy as np
from src.utils.config import DATA_PROCESSED, CONFEDERATION_STRENGTH
from src.utils.logger import get_logger
from src.data.downloader import get_confederation_map

logger = get_logger(__name__)

CONFEDERATION_MAP = get_confederation_map()


def _conf_elo(team: str) -> float:
    conf = CONFEDERATION_MAP.get(team, "UEFA")
    return CONFEDERATION_STRENGTH.get(conf, 1500)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect df to already have Elo and form columns (from elo.py and form.py).
    Returns a DataFrame with engineered features + target.
    """
    feat = pd.DataFrame()

    # --- Elo features ---
    feat["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    feat["home_elo"] = df["home_elo_before"]
    feat["away_elo"] = df["away_elo_before"]

    # --- Confederation strength proxy ---
    feat["home_conf_elo"] = df["home_team"].map(_conf_elo)
    feat["away_conf_elo"] = df["away_team"].map(_conf_elo)
    feat["conf_elo_diff"] = feat["home_conf_elo"] - feat["away_conf_elo"]

    # --- Form features (short & long window) ---
    for w in (5, 10):
        feat[f"home_win_rate_{w}"] = df[f"home_form_{w}_wins"]
        feat[f"away_win_rate_{w}"] = df[f"away_form_{w}_wins"]
        feat[f"home_draw_rate_{w}"] = df[f"home_form_{w}_draws"]
        feat[f"away_draw_rate_{w}"] = df[f"away_form_{w}_draws"]
        feat[f"home_gf_avg_{w}"] = df[f"home_form_{w}_gf"]
        feat[f"away_gf_avg_{w}"] = df[f"away_form_{w}_gf"]
        feat[f"home_ga_avg_{w}"] = df[f"home_form_{w}_ga"]
        feat[f"away_ga_avg_{w}"] = df[f"away_form_{w}_ga"]
        feat[f"home_gd_avg_{w}"] = df[f"home_form_{w}_gd"]
        feat[f"away_gd_avg_{w}"] = df[f"away_form_{w}_gd"]

    feat["win_rate_diff_5"] = feat["home_win_rate_5"] - feat["away_win_rate_5"]
    feat["win_rate_diff_10"] = feat["home_win_rate_10"] - feat["away_win_rate_10"]
    feat["gd_diff_5"] = feat["home_gd_avg_5"] - feat["away_gd_avg_5"]
    feat["gd_diff_10"] = feat["home_gd_avg_10"] - feat["away_gd_avg_10"]

    # --- Attack / defense strength ---
    feat["home_attack_strength"] = df["home_attack_strength"]
    feat["away_attack_strength"] = df["away_attack_strength"]
    feat["home_defense_strength"] = df["home_defense_strength"]
    feat["away_defense_strength"] = df["away_defense_strength"]
    feat["attack_diff"] = feat["home_attack_strength"] - feat["away_attack_strength"]
    feat["defense_diff"] = feat["away_defense_strength"] - feat["home_defense_strength"]

    # --- Contextual flags ---
    feat["neutral_venue"] = df["neutral"].astype(int)
    feat["is_host"] = df.get("is_host", pd.Series(False, index=df.index)).astype(int)
    feat["is_world_cup"] = df["is_world_cup"].astype(int)
    feat["is_knockout"] = df.get("is_knockout", pd.Series(False, index=df.index)).astype(int)

    # --- Meta (kept for slicing, not fed to model) ---
    feat["date"] = df["date"]
    feat["home_team"] = df["home_team"]
    feat["away_team"] = df["away_team"]
    feat["tournament"] = df["tournament"]

    # --- Target ---
    # 2 = home win, 1 = draw, 0 = away win
    feat["target"] = df["outcome"].map({"home_win": 2, "draw": 1, "away_win": 0})

    feat = feat.dropna(subset=["target"]).reset_index(drop=True)
    logger.info(f"Feature matrix: {feat.shape[0]:,} rows × {feat.shape[1]} columns")
    return feat


FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "conf_elo_diff", "home_conf_elo", "away_conf_elo",
    "home_win_rate_5", "away_win_rate_5", "home_draw_rate_5", "away_draw_rate_5",
    "home_win_rate_10", "away_win_rate_10", "home_draw_rate_10", "away_draw_rate_10",
    "home_gf_avg_5", "away_gf_avg_5", "home_ga_avg_5", "away_ga_avg_5",
    "home_gf_avg_10", "away_gf_avg_10", "home_ga_avg_10", "away_ga_avg_10",
    "home_gd_avg_5", "away_gd_avg_5", "home_gd_avg_10", "away_gd_avg_10",
    "win_rate_diff_5", "win_rate_diff_10", "gd_diff_5", "gd_diff_10",
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "attack_diff", "defense_diff",
    "neutral_venue", "is_host", "is_world_cup", "is_knockout",
]


def save_features(feat: pd.DataFrame, filename: str = "features.csv") -> None:
    path = DATA_PROCESSED / filename
    feat.to_csv(path, index=False)
    logger.info(f"Features saved -> {path}")

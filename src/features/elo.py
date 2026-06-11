"""
Compute rolling Elo ratings for every team across the full match history.

Algorithm
---------
  expected_A = 1 / (1 + 10^((elo_B - elo_A) / 400))
  new_elo_A  = elo_A + K * (actual_A - expected_A)

K is adjusted by match importance:
  Friendly           → K = 20
  Qualification      → K = 25
  Continental Cup    → K = 35
  World Cup          → K = 60
  World Cup Final    → K = 75
"""
import pandas as pd
import numpy as np
from src.utils.config import INITIAL_ELO
from src.utils.logger import get_logger

logger = get_logger(__name__)

_K_MAP = {
    "friendly": 20,
    "qualification": 25,
    "continental": 35,
    "world cup": 60,
    "world cup final": 75,
}


def _k_factor(tournament: str) -> int:
    t = tournament.lower()
    if "final" in t and "world cup" in t:
        return _K_MAP["world cup final"]
    if "world cup" in t:
        return _K_MAP["world cup"]
    if any(kw in t for kw in ("euro", "copa", "gold cup", "africa cup", "asian cup", "nations")):
        return _K_MAP["continental"]
    if "qualif" in t:
        return _K_MAP["qualification"]
    return _K_MAP["friendly"]


def compute_elo_ratings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Walk the match history chronologically, update Elo after each game, and
    attach pre-match Elo values as new columns.

    Returns
    -------
    df_with_elo : DataFrame with home_elo_before / away_elo_before columns
    final_elos  : dict mapping team → current Elo rating
    """
    df = df.sort_values("date").reset_index(drop=True)
    elo: dict[str, float] = {}

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        elo.setdefault(home, INITIAL_ELO)
        elo.setdefault(away, INITIAL_ELO)

        home_e = elo[home]
        away_e = elo[away]

        # Store pre-match ratings
        home_elos.append(home_e)
        away_elos.append(away_e)

        # Expected scores
        exp_home = 1 / (1 + 10 ** ((away_e - home_e) / 400))
        exp_away = 1 - exp_home

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if row["home_score"] > row["away_score"]:
            act_home, act_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        k = _k_factor(str(row.get("tournament", "friendly")))
        elo[home] = home_e + k * (act_home - exp_home)
        elo[away] = away_e + k * (act_away - exp_away)

    df = df.copy()
    df["home_elo_before"] = home_elos
    df["away_elo_before"] = away_elos
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]

    logger.info(f"Elo ratings computed for {len(elo)} teams.")
    return df, elo

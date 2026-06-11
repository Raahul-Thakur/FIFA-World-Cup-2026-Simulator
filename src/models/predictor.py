"""
Single-match and batch prediction interface.

Given a trained model and current team stats, predicts:
  P(home_win), P(draw), P(away_win)

Also generates a human-readable natural language explanation.
"""
import numpy as np
import pandas as pd
from src.features.builder import FEATURE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_match(
    model,
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_stats: dict,
    away_stats: dict,
    neutral: bool = True,
    is_world_cup: bool = True,
    is_knockout: bool = False,
    conf_elo_diff: float = 0.0,
) -> dict:
    """
    Build a feature vector for a single match and return probability dict.

    Parameters
    ----------
    home_stats / away_stats : dicts with keys matching form feature suffixes
        e.g. {'win_rate_5': 0.6, 'gd_avg_5': 0.8, 'attack_strength': 1.2, ...}
    """
    row = {col: 0.0 for col in FEATURE_COLS}

    row["elo_diff"] = home_elo - away_elo
    row["home_elo"] = home_elo
    row["away_elo"] = away_elo
    row["conf_elo_diff"] = conf_elo_diff

    # Flatten home/away stats into feature row
    for key, val in home_stats.items():
        col = f"home_{key}"
        if col in row:
            row[col] = val
    for key, val in away_stats.items():
        col = f"away_{key}"
        if col in row:
            row[col] = val

    row["win_rate_diff_5"] = home_stats.get("win_rate_5", 0) - away_stats.get("win_rate_5", 0)
    row["win_rate_diff_10"] = home_stats.get("win_rate_10", 0) - away_stats.get("win_rate_10", 0)
    row["gd_diff_5"] = home_stats.get("gd_avg_5", 0) - away_stats.get("gd_avg_5", 0)
    row["gd_diff_10"] = home_stats.get("gd_avg_10", 0) - away_stats.get("gd_avg_10", 0)
    row["attack_diff"] = home_stats.get("attack_strength", 1) - away_stats.get("attack_strength", 1)
    row["defense_diff"] = away_stats.get("defense_strength", 1) - home_stats.get("defense_strength", 1)

    row["neutral_venue"] = int(neutral)
    row["is_world_cup"] = int(is_world_cup)
    row["is_knockout"] = int(is_knockout)

    X = np.array([[row[c] for c in FEATURE_COLS]])
    probs = model.predict_proba(X)[0]

    # Model classes: [0=away_win, 1=draw, 2=home_win]
    class_order = list(model.classes_)
    prob_map = dict(zip(class_order, probs))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(float(prob_map.get(2, 0)), 4),
        "draw_prob": round(float(prob_map.get(1, 0)), 4),
        "away_win_prob": round(float(prob_map.get(0, 0)), 4),
    }


def explain_prediction(pred: dict, home_elo: float, away_elo: float,
                        home_stats: dict, away_stats: dict) -> str:
    """
    Generate a simple natural-language explanation for the top predicted outcome.
    """
    home = pred["home_team"]
    away = pred["away_team"]
    hw = pred["home_win_prob"]
    dp = pred["draw_prob"]
    aw = pred["away_win_prob"]

    if hw >= aw and hw >= dp:
        winner, win_prob = home, hw
    elif aw >= hw and aw >= dp:
        winner, win_prob = away, aw
    else:
        winner, win_prob = None, dp

    reasons = []
    elo_diff = home_elo - away_elo
    if abs(elo_diff) > 50:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"stronger Elo rating ({abs(elo_diff):.0f} pts ahead)")

    gd_diff = home_stats.get("gd_avg_5", 0) - away_stats.get("gd_avg_5", 0)
    if abs(gd_diff) > 0.3:
        better = home if gd_diff > 0 else away
        reasons.append(f"better recent goal difference (+{abs(gd_diff):.1f} per game for {better})")

    atk_diff = home_stats.get("attack_strength", 1) - away_stats.get("attack_strength", 1)
    if abs(atk_diff) > 0.1:
        better = home if atk_diff > 0 else away
        reasons.append(f"superior attacking form ({better})")

    wr_diff = home_stats.get("win_rate_5", 0) - away_stats.get("win_rate_5", 0)
    if abs(wr_diff) > 0.1:
        better = home if wr_diff > 0 else away
        reasons.append(f"higher recent win rate ({better})")

    if not reasons:
        reasons = ["closely matched statistics"]

    reason_str = ", ".join(reasons) if reasons else "closely matched statistics"

    if winner:
        return (
            f"{winner} has a {win_prob*100:.1f}% chance to beat "
            f"{'the opponent' if winner == home else home} "
            f"because of {reason_str}."
        )
    else:
        return (
            f"This match is likely to end in a draw ({dp*100:.1f}%) "
            f"because of {reason_str}."
        )

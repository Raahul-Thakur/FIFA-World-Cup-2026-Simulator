"""
Advanced features that significantly improve draw prediction and overall accuracy.

New features added
------------------
1. Head-to-head (H2H) stats: win/draw/loss rate and goal stats between the
   specific pair of teams (last 5 H2H meetings).
2. Team draw tendency: each team's historical draw rate over last 20 matches.
3. Elo closeness: non-linear transform that peaks when teams are evenly matched.
4. Days since last match: proxy for fatigue/freshness.
5. Elo momentum: whether Elo is trending up/down over last 10 matches.
6. Absolute Elo level: very strong teams win even close Elo matchups.
"""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _h2h_stats(df: pd.DataFrame, home: str, away: str, before_idx: int, window: int = 5) -> dict:
    """Last `window` meetings between this exact pair (either side)."""
    mask = (
        ((df["home_team"] == home) & (df["away_team"] == away)) |
        ((df["home_team"] == away) & (df["away_team"] == home))
    ) & (df.index < before_idx)

    recent = df[mask].tail(window)
    if recent.empty:
        return {"h2h_home_win_rate": 0.33, "h2h_draw_rate": 0.25, "h2h_away_win_rate": 0.33,
                "h2h_home_gd": 0.0, "h2h_n": 0}

    wins = draws = losses = gd_sum = 0
    for _, r in recent.iterrows():
        if r["home_team"] == home:
            gd = r["home_score"] - r["away_score"]
        else:
            gd = r["away_score"] - r["home_score"]
        gd_sum += gd
        if gd > 0: wins += 1
        elif gd == 0: draws += 1
        else: losses += 1

    n = len(recent)
    return {
        "h2h_home_win_rate": wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_away_win_rate": losses / n,
        "h2h_home_gd": gd_sum / n,
        "h2h_n": n,
    }


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head features for every row.  Slower than vectorised form
    features (requires row-by-row because pairs are unique), but H2H has
    typically only a few matches so the inner loop is cheap.
    """
    logger.info("Computing head-to-head features ...")
    df = df.reset_index(drop=True)

    h2h_cols = {
        "h2h_home_win_rate": [], "h2h_draw_rate": [], "h2h_away_win_rate": [],
        "h2h_home_gd": [], "h2h_n": [],
    }

    for idx, row in df.iterrows():
        stats = _h2h_stats(df, row["home_team"], row["away_team"], idx, window=5)
        for k, v in stats.items():
            h2h_cols[k].append(v)

    for col, vals in h2h_cols.items():
        df[col] = vals

    logger.info("H2H features done.")
    return df


def compute_draw_tendency(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Per-team draw rate over last `window` matches, shifted to avoid leakage.
    Teams like Netherlands or Sweden draw much more often than Brazil — this
    is a strong signal for draw probability.
    """
    logger.info("Computing draw tendency features ...")
    df = df.sort_values("date").reset_index(drop=True)

    home_draw_rows = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "date": df["date"].values,
        "team": df["home_team"].values,
        "draw": (df["home_score"] == df["away_score"]).astype(np.int8).values,
    })
    away_draw_rows = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "date": df["date"].values,
        "team": df["away_team"].values,
        "draw": (df["home_score"] == df["away_score"]).astype(np.int8).values,
    })
    long = pd.concat([home_draw_rows, away_draw_rows], ignore_index=True)
    long = long.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    long["draw_tendency"] = (
        long.groupby("team")["draw"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        .fillna(0.25)
    )

    # Split back — home tendency
    home_long = long[long["team"] == df.loc[long["match_idx"].values, "home_team"].values]

    # Vectorised split using side tags
    original_home = df["home_team"].values
    long["is_home"] = [original_home[idx] == team
                       for idx, team in zip(long["match_idx"], long["team"])]

    home_part = long[long["is_home"]].set_index("match_idx")["draw_tendency"]
    away_part = long[~long["is_home"]].set_index("match_idx")["draw_tendency"]

    df["home_draw_tendency"] = home_part.reindex(range(len(df))).values
    df["away_draw_tendency"] = away_part.reindex(range(len(df))).values
    df["combined_draw_tendency"] = (
        df["home_draw_tendency"] + df["away_draw_tendency"]
    ) / 2

    logger.info("Draw tendency done.")
    return df


def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive non-linear Elo-based features that help predict draws and upsets.
    """
    # Elo closeness: peaks when |elo_diff| is small — strong draw signal
    # Uses a Gaussian kernel centred at 0
    df["elo_closeness"] = np.exp(-(df["elo_diff"] ** 2) / (2 * 150 ** 2))

    # Absolute team level (average Elo) — top teams convert narrow margins into wins
    df["avg_elo"] = (df["home_elo_before"] + df["away_elo_before"]) / 2

    # Elo momentum: is the home/away team's Elo trending up?
    # Proxy: current Elo minus 10-match-ago Elo (positive = improving)
    # We'll compute this from the long form already in df if available
    if "home_elo_before" in df.columns:
        home_elo_trend = df.groupby("home_team")["home_elo_before"].transform(
            lambda s: s.diff(10).fillna(0)
        )
        away_elo_trend = df.groupby("away_team")["away_elo_before"].transform(
            lambda s: s.diff(10).fillna(0)
        )
        df["home_elo_momentum"] = home_elo_trend
        df["away_elo_momentum"] = away_elo_trend
        df["elo_momentum_diff"] = home_elo_trend - away_elo_trend

    return df


def compute_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Days since each team's last match — proxy for fatigue and sharpness.
    Very short rest (<4 days) or very long break (>60 days) both hurt performance.
    """
    logger.info("Computing days-rest features ...")
    df = df.sort_values("date").reset_index(drop=True)

    long = pd.concat([
        pd.DataFrame({"match_idx": np.arange(len(df)), "date": df["date"].values,
                      "team": df["home_team"].values, "side": "home"}),
        pd.DataFrame({"match_idx": np.arange(len(df)), "date": df["date"].values,
                      "team": df["away_team"].values, "side": "away"}),
    ], ignore_index=True).sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    long["days_since_last"] = (
        long.groupby("team")["date"]
        .transform(lambda s: s.diff().dt.days.shift(1))
        .fillna(30)  # neutral default
        .clip(1, 90)
    )

    for side in ("home", "away"):
        side_df = long[long["side"] == side].set_index("match_idx")["days_since_last"]
        df[f"{side}_days_rest"] = side_df.reindex(range(len(df))).fillna(30).values

    df["days_rest_diff"] = df["home_days_rest"] - df["away_days_rest"]

    logger.info("Days-rest done.")
    return df


ADVANCED_FEATURE_COLS = [
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "h2h_home_gd", "h2h_n",
    "home_draw_tendency", "away_draw_tendency", "combined_draw_tendency",
    "elo_closeness", "avg_elo",
    "home_elo_momentum", "away_elo_momentum", "elo_momentum_diff",
    "home_days_rest", "away_days_rest", "days_rest_diff",
]

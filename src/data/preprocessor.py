"""
Clean and standardise raw match data.

Key responsibilities
--------------------
- Parse and sort dates
- Normalise team names (handles minor spelling variants)
- Derive match outcome (home_win / draw / away_win)
- Flag World Cup matches, neutral venues, and host nations
- Attach confederation labels
"""
import pandas as pd
import numpy as np
from src.utils.config import DATA_PROCESSED, WORLD_CUP_YEARS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minor name corrections so lookups are consistent across datasets
NAME_MAP = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "United States": "USA",
    "Czechia": "Czech Republic",
    "Republic of Ireland": "Ireland",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "North Macedonia": "Macedonia",
    "Trinidad and Tobago": "Trinidad & Tobago",
}


def _normalise_name(name: str) -> str:
    return NAME_MAP.get(name, name)


def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw results DataFrame (from downloader) and return a clean version
    ready for feature engineering.
    """
    df = df_raw.copy()

    # --- basic cleaning ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["home_team"] = df["home_team"].map(_normalise_name).fillna(df["home_team"])
    df["away_team"] = df["away_team"].map(_normalise_name).fillna(df["away_team"])

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

    # --- derived columns ---
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["outcome"] = np.select(
        [df["goal_diff"] > 0, df["goal_diff"] < 0],
        ["home_win", "away_win"],
        default="draw",
    )
    df["result_code"] = df["outcome"].map(
        {"home_win": 1, "draw": 0, "away_win": -1}
    ).astype(int)

    # 1 = home wins / draw treated as not-home-win for simplicity later
    df["home_win_flag"] = (df["outcome"] == "home_win").astype(int)
    df["away_win_flag"] = (df["outcome"] == "away_win").astype(int)
    df["draw_flag"] = (df["outcome"] == "draw").astype(int)

    df["year"] = df["date"].dt.year
    df["is_world_cup"] = df["tournament"].str.contains(
        "FIFA World Cup", na=False
    ) & ~df["tournament"].str.contains("qualification", case=False, na=False)

    df["is_knockout"] = df["tournament"].str.contains(
        "World Cup", na=False
    ) & df["tournament"].str.contains(
        "quarter|semi|final|round of", case=False, na=False
    )

    # Neutral venue: already present in the real dataset; if missing default False
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)

    # Host nation: home_team == country where match was played AND it's not neutral
    if "country" in df.columns:
        df["is_host"] = (df["home_team"] == df["country"]) & ~df["neutral"]
    else:
        df["is_host"] = False

    logger.info(f"Clean dataset: {len(df):,} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def filter_modern_era(df: pd.DataFrame, from_year: int = 1993) -> pd.DataFrame:
    """
    Keep only matches from a given year onward.
    1993 is a common cutoff because FIFA adopted the 3-points-for-a-win rule in
    1994, and squad quality and tactical styles differ significantly pre-1990.
    """
    filtered = df[df["year"] >= from_year].copy().reset_index(drop=True)
    logger.info(f"Modern-era subset: {len(filtered):,} matches (>= {from_year})")
    return filtered


def save_processed(df: pd.DataFrame, filename: str = "matches_clean.csv") -> None:
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved -> {path}")

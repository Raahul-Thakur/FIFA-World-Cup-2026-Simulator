"""
Download raw datasets from public sources.

Primary source: Mart Jürisoo's international football results dataset on GitHub
(https://github.com/martj42/international_results) — no API key required.

Fallback synthetic data is generated when the network is unavailable so the
pipeline can still be exercised locally without an internet connection.
"""
import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import DATA_RAW
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/shootouts.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)


def _download_csv(url: str, save_path: Path) -> pd.DataFrame:
    """Download a CSV from url, save locally, and return a DataFrame."""
    if save_path.exists():
        logger.info(f"Cached file found: {save_path.name} — skipping download.")
        return pd.read_csv(save_path)

    logger.info(f"Downloading {save_path.name} …")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")
        return df
    except Exception as exc:
        logger.warning(f"Download failed ({exc}). Generating synthetic data.")
        return None


def download_results() -> pd.DataFrame:
    return _download_csv(RESULTS_URL, DATA_RAW / "results.csv")


def download_shootouts() -> pd.DataFrame:
    return _download_csv(SHOOTOUTS_URL, DATA_RAW / "shootouts.csv")


def download_goalscorers() -> pd.DataFrame:
    return _download_csv(GOALSCORERS_URL, DATA_RAW / "goalscorers.csv")


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------
_TEAMS = [
    "Brazil", "Germany", "Argentina", "France", "Spain", "England",
    "Italy", "Netherlands", "Portugal", "Belgium", "Croatia", "Uruguay",
    "Mexico", "Colombia", "Japan", "South Korea", "Senegal", "Morocco",
    "USA", "Australia", "Switzerland", "Denmark", "Poland", "Serbia",
    "Ghana", "Cameroon", "Ecuador", "Wales", "Qatar", "Canada",
]

_CONFEDERATIONS = {
    "Brazil": "CONMEBOL", "Germany": "UEFA", "Argentina": "CONMEBOL",
    "France": "UEFA", "Spain": "UEFA", "England": "UEFA", "Italy": "UEFA",
    "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Uruguay": "CONMEBOL", "Mexico": "CONCACAF",
    "Colombia": "CONMEBOL", "Japan": "AFC", "South Korea": "AFC",
    "Senegal": "CAF", "Morocco": "CAF", "USA": "CONCACAF",
    "Australia": "AFC", "Switzerland": "UEFA", "Denmark": "UEFA",
    "Poland": "UEFA", "Serbia": "UEFA", "Ghana": "CAF",
    "Cameroon": "CAF", "Ecuador": "CONMEBOL", "Wales": "UEFA",
    "Qatar": "AFC", "Canada": "CONCACAF",
}


def generate_synthetic_data(n_matches: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic international match data when the
    real dataset cannot be fetched.  Elo-based win probabilities drive the
    simulated scores so the resulting features are internally consistent.
    """
    rng = np.random.default_rng(seed)
    teams = _TEAMS

    # Assign base Elo ratings that reflect rough historical strength
    base_elo = {t: 1500 + rng.integers(-200, 201) for t in teams}

    tournaments = [
        "FIFA World Cup", "FIFA World Cup qualification", "Friendly",
        "UEFA Euro", "Copa América", "Africa Cup of Nations",
        "AFC Asian Cup", "CONCACAF Gold Cup",
    ]
    tournament_weights = [0.08, 0.25, 0.30, 0.10, 0.08, 0.07, 0.06, 0.06]

    rows = []
    date = pd.Timestamp("1993-01-01")
    for _ in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        tournament = rng.choice(tournaments, p=tournament_weights)
        neutral = bool(rng.integers(0, 2))

        elo_diff = base_elo[home] - base_elo[away]
        if not neutral:
            elo_diff += 100  # home advantage

        # Convert Elo diff to win probability (logistic)
        home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        draw_prob = 0.25
        home_win_prob = max(0.05, home_win_prob - draw_prob / 2)
        away_win_prob = max(0.05, 1 - home_win_prob - draw_prob)

        outcome = rng.choice(
            ["home", "draw", "away"],
            p=[home_win_prob, draw_prob, away_win_prob],
        )

        base_goals = max(0, int(rng.normal(1.3, 1.0)))
        if outcome == "home":
            hs, as_ = base_goals + rng.integers(1, 3), max(0, base_goals - 1)
        elif outcome == "away":
            hs, as_ = max(0, base_goals - 1), base_goals + rng.integers(1, 3)
        else:
            g = base_goals
            hs, as_ = g, g

        # Nudge Elo after the match
        expected = 1 / (1 + 10 ** ((base_elo[away] - base_elo[home]) / 400))
        actual = 1.0 if outcome == "home" else (0.5 if outcome == "draw" else 0.0)
        base_elo[home] += 20 * (actual - expected)
        base_elo[away] += 20 * ((1 - actual) - (1 - expected))

        date += pd.Timedelta(days=int(rng.integers(1, 15)))
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": int(hs),
            "away_score": int(as_),
            "tournament": tournament,
            "city": "Various",
            "country": away if neutral else home,
            "neutral": neutral,
        })

    df = pd.DataFrame(rows)
    save_path = DATA_RAW / "results.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic dataset with {len(df)} matches saved to {save_path}")
    return df


def load_or_generate_results() -> pd.DataFrame:
    """Try to download real data; fall back to synthetic if unavailable."""
    df = download_results()
    if df is None or df.empty:
        logger.warning("Using synthetic match data.")
        df = generate_synthetic_data()
    return df


def get_confederation_map() -> dict:
    return _CONFEDERATIONS.copy()

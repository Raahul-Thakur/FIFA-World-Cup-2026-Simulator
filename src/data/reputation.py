"""
Pre-tournament *reputation* prior for World Cup 2026 teams.

The historical Elo ratings in `current_elos.csv` are compressed — every contender
sits in a narrow 1850-1990 band, so a few in-form sides (Morocco, Japan, Ecuador)
edge ahead of traditional powers (Brazil, Netherlands) by a handful of points,
turning genuine favourites into coin-flips.

To de-compress, we blend each team's data Elo with a curated reputation rating
that reflects broad 2025-26 consensus strength tiers (think FIFA-ranking / seeding
pots, spread across ~1650-2110).  The blend keeps recent form mattering while
restoring a realistic gap between tiers.

    adjusted = W_DATA * data_elo + (1 - W_DATA) * reputation

Teams without a reputation entry (e.g. obscure sides in the custom-match tab) fall
back to their data Elo unchanged.
"""
from __future__ import annotations

# Weight on the data-driven Elo vs the reputation prior (0..1).
W_DATA = 0.5

# Reputation ratings — consensus pre-tournament strength, names matching wc2026.py.
REPUTATION_ELO = {
    # Elite
    "Argentina": 2110, "France": 2085, "Spain": 2080,
    # Top
    "Brazil": 2030, "England": 2015, "Portugal": 2005,
    "Netherlands": 1990, "Germany": 1980, "Belgium": 1955,
    # Strong
    "Croatia": 1930, "Uruguay": 1925, "Colombia": 1905, "Morocco": 1900,
    "Japan": 1895, "Senegal": 1890, "Switzerland": 1880, "USA": 1880,
    "Norway": 1880, "Mexico": 1875,
    # Upper-mid
    "Turkey": 1870, "Ecuador": 1860, "Austria": 1860, "South Korea": 1855,
    "Czechia": 1850, "Ivory Coast": 1845, "Canada": 1845, "Sweden": 1845,
    "Egypt": 1840, "Iran": 1840, "Algeria": 1835, "Bosnia and Herzegovina": 1830,
    "Scotland": 1825, "Ghana": 1820, "Paraguay": 1820, "Australia": 1810,
    # Mid
    "DR Congo": 1795, "Tunisia": 1795, "South Africa": 1780, "Qatar": 1760,
    "Saudi Arabia": 1760, "Uzbekistan": 1755, "Cape Verde": 1730, "Panama": 1730,
    # Lower
    "Iraq": 1730, "Jordan": 1715, "New Zealand": 1700, "Haiti": 1690,
    "Curacao": 1680,
}


def adjusted_elo(team: str, data_elo: float, w_data: float = W_DATA) -> float:
    """Blend a team's data Elo with its reputation prior (if one exists)."""
    rep = REPUTATION_ELO.get(team)
    if rep is None:
        return data_elo
    return w_data * data_elo + (1.0 - w_data) * rep

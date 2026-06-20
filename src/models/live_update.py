"""
In-tournament update engine.

Given the actual scorelines of completed 2026 group-stage matches, this module:

1.  Updates each team's Elo rating using the same World-Cup Elo step the project's
    history uses (K = 60, expected = 1/(1+10**((opp-team)/400)), no goal multiplier).
2.  Builds the live group standings (played / W / D / L / GF / GA / GD / points)
    from the actual results only.
3.  Splits the fixture list into played vs remaining at a chosen cut-off match.

Results are keyed by (home_display, away_display) — the display names used in
`wc2026.FIXTURES` — mapping to a (home_goals, away_goals) tuple.
"""
from __future__ import annotations

from collections import defaultdict

from src.data.wc2026 import FIXTURES, GROUPS, elo_name

WC_K = 60  # World-Cup K-factor, matching src/features/elo.py


def ordered_fixtures() -> list:
    """All fixtures in true chronological order (date, then kickoff time)."""
    return sorted(FIXTURES, key=lambda f: (f["date"], f.get("time", "")))


def split_fixtures(cut_home: str = "Turkey", cut_away: str = "Paraguay") -> tuple[list, list]:
    """Return (played, remaining) split *inclusive* of the cut-off fixture."""
    ordered = ordered_fixtures()
    idx = next(
        (i for i, f in enumerate(ordered) if f["home"] == cut_home and f["away"] == cut_away),
        len(ordered) - 1,
    )
    return ordered[: idx + 1], ordered[idx + 1:]


def _elo_step(team_e: float, opp_e: float, actual: float, k: int = WC_K) -> float:
    expected = 1.0 / (1.0 + 10 ** ((opp_e - team_e) / 400.0))
    return team_e + k * (actual - expected)


def apply_results_to_elo(base_elos: dict, results: dict, k: int = WC_K) -> dict:
    """
    Walk the played fixtures in chronological order and update Elo from results.

    `base_elos` is keyed by data-table names (current_elos.csv). `results` is keyed
    by (home_display, away_display) → (home_goals, away_goals). Returns a new
    data-name-keyed dict (input left untouched).
    """
    elo = dict(base_elos)
    for f in ordered_fixtures():
        key = (f["home"], f["away"])
        if key not in results:
            continue
        hs, as_ = results[key]
        if hs is None or as_ is None:
            continue
        hn, an = elo_name(f["home"]), elo_name(f["away"])
        he = elo.get(hn, 1500.0)
        ae = elo.get(an, 1500.0)
        if hs > as_:
            act_h, act_a = 1.0, 0.0
        elif hs < as_:
            act_h, act_a = 0.0, 1.0
        else:
            act_h, act_a = 0.5, 0.5
        elo[hn] = _elo_step(he, ae, act_h, k)
        elo[an] = _elo_step(ae, he, act_a, k)
    return elo


def group_standings(results: dict) -> dict:
    """
    Live group tables from actual results.

    Returns {group: [row, ...]} where each row is a dict with played/W/D/L/
    GF/GA/GD/Pts, sorted by Pts → GD → GF (head-to-head omitted).
    """
    stats = {
        team: {"team": team, "played": 0, "W": 0, "D": 0, "L": 0,
               "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for members in GROUPS.values() for team in members
    }
    for f in FIXTURES:
        key = (f["home"], f["away"])
        if key not in results:
            continue
        hs, as_ = results[key]
        if hs is None or as_ is None:
            continue
        h, a = f["home"], f["away"]
        for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
            s = stats[t]
            s["played"] += 1
            s["GF"] += gf
            s["GA"] += ga
            s["GD"] += gf - ga
        if hs > as_:
            stats[h]["W"] += 1; stats[h]["Pts"] += 3; stats[a]["L"] += 1
        elif hs < as_:
            stats[a]["W"] += 1; stats[a]["Pts"] += 3; stats[h]["L"] += 1
        else:
            stats[h]["D"] += 1; stats[a]["D"] += 1
            stats[h]["Pts"] += 1; stats[a]["Pts"] += 1

    out = {}
    for g, members in GROUPS.items():
        rows = [stats[t] for t in members]
        rows.sort(key=lambda s: (s["Pts"], s["GD"], s["GF"]), reverse=True)
        out[g] = rows
    return out

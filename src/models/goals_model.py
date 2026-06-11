"""
Expected-goals (xG) and goalscorer model for World Cup 2026 fixtures.

Two complementary layers sit on top of the trained outcome classifier:

1.  Elo -> expected goals.  A team's attacking supremacy scales with its Elo edge;
    independent Poisson distributions over each team's goal count then give the
    most-likely scoreline and a goals-based outcome cross-check.
2.  Expected goals -> goalscorers.  A team's expected goals are split across its
    known scorers using the relative scoring shares in `squads_2026`, and each
    player's chance of scoring at least once is 1 - exp(-player_xG) (Poisson).

These are deliberately simple, transparent priors — a portfolio toy, not a betting
model.
"""
from __future__ import annotations

import math

from src.data.squads_2026 import get_scorer_shares

# Average goals per team in an evenly-matched international (total ≈ 2.7).
BASE_TEAM_GOALS = 1.35
# Multiplicative Elo→goals model: a team's expected goals scale by
# 10**(elo_edge / ELO_GOAL_SCALE).  Smaller scale ⇒ favourites pull further
# ahead.  At 800, a 200-pt edge ≈ ×1.78 goals (and the underdog ÷1.78), a
# 300-pt edge ≈ ×2.4 — so big mismatches produce realistic 3-0 / 4-1 lines
# instead of everything collapsing to 1-0.
ELO_GOAL_SCALE = 800.0
# Home bump for a host nation, expressed as an Elo-equivalent edge.
HOME_FIELD_ELO = 65.0
# Clip the Elo edge and the resulting means to sane football ranges.
MAX_ELO_EDGE = 400.0
MIN_XG, MAX_XG = 0.15, 4.5


def expected_goals(home_elo: float, away_elo: float, neutral: bool = True) -> tuple[float, float]:
    """Return (home_xG, away_xG) means from a multiplicative Elo-goals model."""
    edge = (home_elo - away_elo) + (0.0 if neutral else HOME_FIELD_ELO)
    edge = max(-MAX_ELO_EDGE, min(MAX_ELO_EDGE, edge))
    factor = 10.0 ** (edge / ELO_GOAL_SCALE)
    home_xg = BASE_TEAM_GOALS * factor
    away_xg = BASE_TEAM_GOALS / factor
    return (
        min(MAX_XG, max(MIN_XG, home_xg)),
        min(MAX_XG, max(MIN_XG, away_xg)),
    )


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


# Dixon-Coles low-score dependence parameter.  Real football goals are mildly
# correlated, so independent Poisson under-counts 0-0 and 1-1.  A negative rho
# inflates those draws and trims 1-0 / 0-1.  -0.13 is a typical fitted value.
DIXON_COLES_RHO = -0.13


def _dc_tau(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for the four low-score cells (else 1)."""
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 10,
                 rho: float = DIXON_COLES_RHO) -> dict:
    """Normalised P(home=h, away=a) grid under a Dixon-Coles adjusted Poisson."""
    grid = {}
    total = 0.0
    for h in range(max_goals + 1):
        ph = _poisson_pmf(h, home_xg)
        for a in range(max_goals + 1):
            p = _dc_tau(h, a, home_xg, away_xg, rho) * ph * _poisson_pmf(a, away_xg)
            p = max(p, 0.0)
            grid[(h, a)] = p
            total += p
    if total > 0:
        for k in grid:
            grid[k] /= total
    return grid


def scoreline_distribution(home_xg: float, away_xg: float, max_goals: int = 10) -> dict:
    """
    Dixon-Coles adjusted Poisson scoreline model.

    Returns the modal scoreline plus goals-based outcome probabilities, used as a
    transparent cross-check alongside the trained classifier.
    """
    grid = score_matrix(home_xg, away_xg, max_goals)
    p_home = p_draw = p_away = 0.0
    best_score, best_p = (0, 0), -1.0
    # Also track the most-likely scoreline *within* each outcome, so a displayed
    # scoreline can be made consistent with the predicted W/D/L verdict.
    best_by_outcome = {"home": ((1, 0), -1.0), "draw": ((0, 0), -1.0), "away": ((0, 1), -1.0)}
    for (h, a), p in grid.items():
        if p > best_p:
            best_p, best_score = p, (h, a)
        key = "home" if h > a else ("draw" if h == a else "away")
        if p > best_by_outcome[key][1]:
            best_by_outcome[key] = ((h, a), p)
        if h > a:
            p_home += p
        elif h == a:
            p_draw += p
        else:
            p_away += p
    return {
        "likely_score": best_score,
        "likely_score_prob": best_p,
        "best_home_score": best_by_outcome["home"][0],
        "best_draw_score": best_by_outcome["draw"][0],
        "best_away_score": best_by_outcome["away"][0],
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "exp_total_goals": home_xg + away_xg,
    }


def match_markets(home_xg: float, away_xg: float, top_n: int = 6) -> dict:
    """
    Derived betting-style markets from the scoreline grid.

    Returns the top-N most-likely exact scorelines plus P(draw), P(over 2.5),
    P(over 3.5), and P(both teams score).
    """
    grid = score_matrix(home_xg, away_xg)
    p_draw = p_o25 = p_o35 = p_btts = 0.0
    for (h, a), p in grid.items():
        if h == a:
            p_draw += p
        if h + a >= 3:
            p_o25 += p
        if h + a >= 4:
            p_o35 += p
        if h >= 1 and a >= 1:
            p_btts += p
    top = sorted(grid.items(), key=lambda kv: -kv[1])[:top_n]
    return {
        "top_scores": [{"score": f"{h}-{a}", "prob": round(p, 4)} for (h, a), p in top],
        "p_draw": round(p_draw, 4),
        "p_over25": round(p_o25, 4),
        "p_over35": round(p_o35, 4),
        "p_btts": round(p_btts, 4),
    }


def predict_scorers(team: str, team_xg: float, max_players: int = 4) -> list[dict]:
    """
    Split a team's expected goals across its likely scorers.

    Returns a list of {player, exp_goals, prob_score} sorted by expected goals,
    where prob_score = P(player scores >= 1) under a Poisson(player_xG).
    """
    shares = get_scorer_shares(team)
    out = []
    for player, share in shares[:max_players]:
        player_xg = team_xg * share
        out.append({
            "player": player,
            "exp_goals": round(player_xg, 2),
            "prob_score": round(1.0 - math.exp(-player_xg), 3),
        })
    out.sort(key=lambda d: d["exp_goals"], reverse=True)
    return out


# When the top two outcomes are within this probability margin, the match is
# called a toss-up rather than a confident win for the marginally-higher side.
TOSSUP_MARGIN = 0.07


def predict_fixture(
    home: str,
    away: str,
    home_elo: float,
    away_elo: float,
    neutral: bool = True,
    outcome_probs: dict | None = None,
) -> dict:
    """
    Full fixture prediction: outcome (W/D/L), scoreline, total goals, scorers.

    `outcome_probs`, when supplied (from the trained classifier via
    `predictor.predict_match`), provides the headline W/D/L probabilities; the
    Poisson layer always provides the scoreline, expected goals, and scorers.
    """
    home_xg, away_xg = expected_goals(home_elo, away_elo, neutral=neutral)
    dist = scoreline_distribution(home_xg, away_xg)

    if outcome_probs:
        p_home = outcome_probs.get("home_win_prob", dist["p_home"])
        p_draw = outcome_probs.get("draw_prob", dist["p_draw"])
        p_away = outcome_probs.get("away_win_prob", dist["p_away"])
        outcome_source = "model"
    else:
        p_home, p_draw, p_away = dist["p_home"], dist["p_draw"], dist["p_away"]
        outcome_source = "poisson"

    if p_home >= p_draw and p_home >= p_away:
        verdict = f"{home} win"
        likely_score = dist["best_home_score"]
    elif p_away >= p_home and p_away >= p_draw:
        verdict = f"{away} win"
        likely_score = dist["best_away_score"]
    else:
        verdict = "Draw"
        likely_score = dist["best_draw_score"]

    # Toss-up: top two outcomes within a hair — don't over-state a confident call.
    ranked = sorted([p_home, p_draw, p_away], reverse=True)
    tossup = (ranked[0] - ranked[1]) < TOSSUP_MARGIN
    if tossup:
        verdict = f"Too close to call (lean {verdict})"

    return {
        "home": home,
        "away": away,
        "p_home": round(float(p_home), 4),
        "p_draw": round(float(p_draw), 4),
        "p_away": round(float(p_away), 4),
        "outcome_source": outcome_source,
        "verdict": verdict,
        "tossup": tossup,
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "likely_score": likely_score,
        "exp_total_goals": round(dist["exp_total_goals"], 2),
        "markets": match_markets(home_xg, away_xg),
        "home_scorers": predict_scorers(home, home_xg),
        "away_scorers": predict_scorers(away, away_xg),
    }

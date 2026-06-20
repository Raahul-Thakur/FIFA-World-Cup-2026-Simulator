"""
Monte Carlo World Cup Tournament Simulator.

Simulates the 2026 FIFA World Cup format (48 teams, 12 groups of 4).
Falls back to the classic 32-team / 8-group format when fewer teams are given.

Group stage  : round-robin within each group
               3 pts win / 1 pt draw / 0 pts loss
               Tiebreaker: GD → GF → head-to-head → coin flip
Knockout     : single elimination; drawn matches go to penalties (50/50)
Simulation   : N_SIMULATIONS independent Monte Carlo runs
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.utils.config import N_SIMULATIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 2026 World Cup groups (placeholder; update when official draw is done)
GROUPS_2026 = {
    "A": ["USA", "Mexico", "Canada", "Qatar"],
    "B": ["Brazil", "Colombia", "Ecuador", "Uruguay"],
    "C": ["Argentina", "Chile", "Peru", "Paraguay"],
    "D": ["France", "Belgium", "Switzerland", "Netherlands"],
    "E": ["England", "Germany", "Denmark", "Poland"],
    "F": ["Spain", "Portugal", "Croatia", "Serbia"],
    "G": ["Italy", "Japan", "South Korea", "Australia"],
    "H": ["Morocco", "Senegal", "Ghana", "Cameroon"],
    "I": ["Iran", "Saudi Arabia", "Australia", "South Korea"],
    "J": ["Nigeria", "Algeria", "Tunisia", "Egypt"],
    "K": ["Costa Rica", "Honduras", "Jamaica", "El Salvador"],
    "L": ["Wales", "Slovakia", "Hungary", "Ireland"],
}

# Use the 8-group 32-team format as the fallback
GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "USA", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}


def _win_prob(elo_a: float, elo_b: float, neutral: bool = True) -> float:
    """Expected win probability for team A vs B using Elo."""
    advantage = 0 if neutral else 100
    return 1 / (1 + 10 ** ((elo_b - elo_a - advantage) / 400))


def _simulate_match(
    team_a: str, team_b: str, elo: dict, neutral: bool = True, rng=None
) -> tuple[str, str, int, int]:
    """
    Simulate a single match.

    Returns (winner_or_draw, team_a_goals, team_b_goals).
    Goals are Poisson-drawn from team strengths; outcome is anchored to the
    pre-computed win probability so calibration stays realistic.
    """
    if rng is None:
        rng = np.random.default_rng()

    ea = elo.get(team_a, 1500)
    eb = elo.get(team_b, 1500)

    p_a = _win_prob(ea, eb, neutral)
    p_draw = 0.25
    p_a = max(0.05, p_a - p_draw / 2)
    p_b = max(0.05, 1 - p_a - p_draw)
    # Re-normalise after clipping
    total = p_a + p_draw + p_b
    p_a /= total; p_draw /= total; p_b /= total

    outcome = rng.choice(["a", "draw", "b"], p=[p_a, p_draw, p_b])

    # Simulate realistic scorelines
    avg_goals = 1.35
    if outcome == "a":
        ga = max(1, rng.poisson(avg_goals * 1.2))
        gb = rng.poisson(avg_goals * 0.7)
        if gb >= ga:
            gb = ga - 1
    elif outcome == "b":
        gb = max(1, rng.poisson(avg_goals * 1.2))
        ga = rng.poisson(avg_goals * 0.7)
        if ga >= gb:
            ga = gb - 1
    else:
        g = rng.poisson(avg_goals)
        ga, gb = g, g

    return outcome, int(ga), int(gb)


def _simulate_group(teams: list, elo: dict, rng) -> list:
    """
    Simulate a round-robin group.

    Returns teams sorted by: pts → gd → gf → random tiebreak (as in FIFA rules).
    """
    pts = defaultdict(int)
    gd = defaultdict(int)
    gf = defaultdict(int)

    for i, ta in enumerate(teams):
        for tb in teams[i + 1:]:
            outcome, ga, gb = _simulate_match(ta, tb, elo, neutral=True, rng=rng)
            gf[ta] += ga; gf[tb] += gb
            gd[ta] += ga - gb; gd[tb] += gb - ga
            if outcome == "a":
                pts[ta] += 3
            elif outcome == "b":
                pts[tb] += 3
            else:
                pts[ta] += 1; pts[tb] += 1

    ranking_key = {
        t: (pts[t], gd[t], gf[t], rng.random()) for t in teams
    }
    return sorted(teams, key=lambda t: ranking_key[t], reverse=True)


def _simulate_knockout_match(
    team_a: str, team_b: str, elo: dict, rng
) -> str:
    """Knockout match — penalties (50/50) break draws."""
    outcome, _, _ = _simulate_match(team_a, team_b, elo, neutral=True, rng=rng)
    if outcome == "a":
        return team_a
    elif outcome == "b":
        return team_b
    else:
        return team_a if rng.random() < 0.5 else team_b


def _simulate_tournament(groups: dict, elo: dict, rng) -> dict:
    """
    Full single-tournament simulation.

    Returns dict: stage → set of teams that reached that stage.
    """
    n_groups = len(groups)
    group_order = sorted(groups.keys())
    all_teams = [t for g in groups.values() for t in g]
    stages = {t: "Group" for t in all_teams}

    # --- Group stage ---
    qualified = []  # list of teams that advance
    group_results = {}
    for g in group_order:
        ranked = _simulate_group(groups[g], elo, rng)
        group_results[g] = ranked
        # Top 2 advance in 8-group format; top 2 + best 3rd in 12-group format
        qualified.extend(ranked[:2])

    # Best third-place teams (applicable for 12-group formats)
    if n_groups == 12:
        thirds = [group_results[g][2] for g in group_order]
        # Pick 8 best thirds by Elo (simplified; official uses GD/GF)
        thirds_sorted = sorted(thirds, key=lambda t: elo.get(t, 1500), reverse=True)
        qualified.extend(thirds_sorted[:8])

    for t in qualified:
        stages[t] = "Round of 16"

    # --- Knockout rounds ---
    round_names = ["Quarter-final", "Semi-final", "Final", "Winner"]
    current_round = qualified[:]
    round_idx = 0

    while len(current_round) > 1:
        next_round = []
        rng.shuffle(current_round)  # randomise bracket seeding for simplicity
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                winner = _simulate_knockout_match(
                    current_round[i], current_round[i + 1], elo, rng
                )
                next_round.append(winner)
                if round_idx < len(round_names):
                    stages[winner] = round_names[round_idx]
            else:
                next_round.append(current_round[i])

        current_round = next_round
        round_idx += 1

    if current_round:
        stages[current_round[0]] = "Winner"

    return stages


def run_simulation(
    groups: dict = None,
    elo: dict = None,
    n_simulations: int = N_SIMULATIONS,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation and return a DataFrame of probabilities per team.

    Columns
    -------
    team, group, p_r16, p_qf, p_sf, p_final, p_winner
    """
    if groups is None:
        groups = GROUPS_2022
    if elo is None:
        # Fallback: assign Elo by confederation proxy
        from src.utils.config import CONFEDERATION_STRENGTH
        from src.data.downloader import get_confederation_map
        conf_map = get_confederation_map()
        elo = {
            t: CONFEDERATION_STRENGTH.get(conf_map.get(t, "UEFA"), 1500)
            for t in [t for g in groups.values() for t in g]
        }

    stage_counts = defaultdict(lambda: defaultdict(int))
    rng = np.random.default_rng(seed)

    logger.info(f"Running {n_simulations:,} simulations …")
    for _ in tqdm(range(n_simulations), desc="Simulating", unit="run"):
        result = _simulate_tournament(groups, elo, rng)
        for team, stage in result.items():
            stage_counts[team][stage] += 1

    stage_order = ["Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]

    rows = []
    for g_name, members in groups.items():
        for team in members:
            counts = stage_counts[team]
            total = n_simulations

            # cumulative: "reached at least this stage"
            reached = {s: 0 for s in stage_order}
            cum = 0
            for s in reversed(stage_order):
                cum += counts.get(s, 0)
                reached[s] = cum

            rows.append({
                "team": team,
                "group": g_name,
                "elo": elo.get(team, 1500),
                "p_r16": reached["Round of 16"] / total,
                "p_qf": reached["Quarter-final"] / total,
                "p_sf": reached["Semi-final"] / total,
                "p_final": reached["Final"] / total,
                "p_winner": reached["Winner"] / total,
            })

    df = pd.DataFrame(rows).sort_values("p_winner", ascending=False).reset_index(drop=True)
    logger.info("Simulation complete.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2026-specific simulation: real groups + reputation prior, with group standings
# and Round-of-32 qualification (top 2 per group + 8 best third-placed teams).
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_group_2026(teams: list, elo: dict, rng, fixed: dict = None) -> list:
    """Round-robin group → standings as list of dicts with pts/gd/gf, ranked.

    Ranking: points → goal difference → goals for → random tiebreak (FIFA order,
    head-to-head omitted for simplicity).  `fixed` maps an ordered (ta, tb) pair
    to a known (ta_goals, tb_goals) result so already-played matches are used
    verbatim instead of simulated.
    """
    fixed = fixed or {}
    pts = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}

    for i, ta in enumerate(teams):
        for tb in teams[i + 1:]:
            if (ta, tb) in fixed:
                ga, gb = fixed[(ta, tb)]
                outcome = "a" if ga > gb else ("b" if gb > ga else "draw")
            else:
                outcome, ga, gb = _simulate_match(ta, tb, elo, neutral=True, rng=rng)
            gf[ta] += ga; gf[tb] += gb
            gd[ta] += ga - gb; gd[tb] += gb - ga
            if outcome == "a":
                pts[ta] += 3
            elif outcome == "b":
                pts[tb] += 3
            else:
                pts[ta] += 1; pts[tb] += 1

    ranked = sorted(
        teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True
    )
    return [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in ranked]


def run_wc2026_simulation(data_elos: dict, n_simulations: int = 5000, seed: int = 0,
                          played_results: dict = None) -> pd.DataFrame:
    """
    Monte Carlo over the real 2026 World Cup (48 teams, 12 groups), using each
    team's reputation-adjusted Elo.

    Qualification: group winners + runners-up (24) plus the 8 best third-placed
    teams advance to the Round of 32, then single-elimination to the Final.

    `played_results`, when given, maps (home_display, away_display) → (hg, ag) for
    already-completed group matches; those are used verbatim and only the rest of
    the group stage is simulated (knockouts always simulated).

    Returns one row per team with:
      team, group, elo,
      p_pos1..p_pos4   — probability of finishing 1st/2nd/3rd/4th in the group
      p_ro32           — probability of reaching the Round of 32 (= qualifying)
      p_r16, p_qf, p_sf, p_final, p_winner — knockout stage reach probabilities
    """
    from src.data.wc2026 import GROUPS, elo_name
    from src.data.reputation import adjusted_elo

    group_order = sorted(GROUPS.keys())
    elo = {
        team: adjusted_elo(team, data_elos.get(elo_name(team), 1500))
        for g in GROUPS.values() for team in g
    }

    # Pre-built per-group lookup of fixed (already-played) results, both orientations.
    fixed_by_group = defaultdict(dict)
    team_group = {team: g for g, members in GROUPS.items() for team in members}
    if played_results:
        for (home, away), score in played_results.items():
            if score is None or score[0] is None or score[1] is None:
                continue
            g = team_group.get(home)
            if g is None or team_group.get(away) != g:
                continue  # only group-stage pairs
            hg, ag = score
            fixed_by_group[g][(home, away)] = (hg, ag)
            fixed_by_group[g][(away, home)] = (ag, hg)

    ko_rounds = ["Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
    pos_counts = defaultdict(lambda: [0, 0, 0, 0])   # team -> [p1,p2,p3,p4]
    ro32_counts = defaultdict(int)
    stage_counts = defaultdict(lambda: defaultdict(int))  # team -> stage -> n

    rng = np.random.default_rng(seed)
    logger.info(f"Running {n_simulations:,} 2026 simulations …")
    for _ in tqdm(range(n_simulations), desc="Simulating 2026", unit="run"):
        thirds = []
        qualifiers = []
        for g in group_order:
            standings = _simulate_group_2026(GROUPS[g], elo, rng, fixed=fixed_by_group.get(g))
            for pos, row in enumerate(standings):
                pos_counts[row["team"]][pos] += 1
            qualifiers.extend([standings[0]["team"], standings[1]["team"]])
            thirds.append(standings[2])

        # 8 best third-placed teams by pts → gd → gf
        thirds.sort(key=lambda r: (r["pts"], r["gd"], r["gf"], rng.random()), reverse=True)
        qualifiers.extend([r["team"] for r in thirds[:8]])

        for t in qualifiers:
            ro32_counts[t] += 1

        # Single-elimination knockout from the 32 qualifiers
        current = qualifiers[:]
        rng.shuffle(current)
        round_idx = 0
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                winner = _simulate_knockout_match(current[i], current[i + 1], elo, rng)
                nxt.append(winner)
                if round_idx < len(ko_rounds):
                    stage_counts[winner][ko_rounds[round_idx]] += 1
            current = nxt
            round_idx += 1

    rows = []
    total = n_simulations
    for g in group_order:
        for team in GROUPS[g]:
            pc = pos_counts[team]
            sc = stage_counts[team]
            # cumulative "reached at least this knockout stage"
            reached = {}
            cum = 0
            for s in reversed(ko_rounds):
                cum += sc.get(s, 0)
                reached[s] = cum
            rows.append({
                "team": team,
                "group": g,
                "elo": round(elo[team], 1),
                "p_pos1": pc[0] / total,
                "p_pos2": pc[1] / total,
                "p_pos3": pc[2] / total,
                "p_pos4": pc[3] / total,
                "p_ro32": ro32_counts[team] / total,
                "p_r16": reached["Round of 16"] / total,
                "p_qf": reached["Quarter-final"] / total,
                "p_sf": reached["Semi-final"] / total,
                "p_final": reached["Final"] / total,
                "p_winner": reached["Winner"] / total,
            })

    df = pd.DataFrame(rows).sort_values("p_winner", ascending=False).reset_index(drop=True)
    logger.info("2026 simulation complete.")
    return df

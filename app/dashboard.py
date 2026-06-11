"""
FIFA World Cup Predictor — Streamlit Dashboard

Run with:  streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helper: load artefacts lazily ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_sim_results():
    p = Path("data/processed/simulation_results.csv")
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_features():
    p = Path("data/processed/features.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return None


@st.cache_data(show_spinner=False)
def load_model_results():
    p = Path("reports/model_results.csv")
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return None


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str = "xgboost"):
    import joblib
    for name in [model_name, "lightgbm", "random_forest", "logistic_regression"]:
        p = Path(f"models_saved/{name}.joblib")
        if p.exists():
            return joblib.load(p), name
    return None, None


@st.cache_data(show_spinner=False)
def load_current_elos():
    p = Path("data/processed/current_elos.csv")
    if p.exists():
        df = pd.read_csv(p, index_col="team")
        return df["elo"].to_dict()
    return {}


def _data_rev() -> str:
    """Short hash of the prediction inputs.

    Passed into the cached compute functions as a key, so `st.cache_data` busts
    automatically whenever the reputation prior, scorer table, or goals-model
    constants change — `st.cache_data` otherwise only notices edits to a cached
    function's *own* code, not to the data modules it imports. Bump `logic` for
    pure-algorithm changes that don't touch any of the values below.
    """
    import hashlib, json
    from src.data.reputation import REPUTATION_ELO, W_DATA
    from src.data.squads_2026 import SCORER_SHARES
    from src.models import goals_model as gm
    payload = {
        "rep": REPUTATION_ELO, "w": W_DATA,
        "shares": {k: [[p, round(w, 4)] for p, w in v] for k, v in SCORER_SHARES.items()},
        "gm": [gm.BASE_TEAM_GOALS, gm.ELO_GOAL_SCALE, gm.HOME_FIELD_ELO,
               gm.MAX_ELO_EDGE, gm.TOSSUP_MARGIN, gm.DIXON_COLES_RHO],
        "logic": 3,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def _elo_stats(elo: float) -> dict:
    """Synthetic per-team form stats derived from an Elo rating.

    Mirrors the assumptions used in the single-match predictor so the schedule
    view and the custom-match view stay consistent.
    """
    return {
        "win_rate_5": 0.5, "win_rate_10": 0.5,
        "draw_rate_5": 0.25, "draw_rate_10": 0.25,
        "gf_avg_5": 1.4, "ga_avg_5": 1.0,
        "gd_avg_5": 0.0, "gd_avg_10": 0.0,
        "attack_strength": elo / 1500,
        "defense_strength": 1500 / max(elo, 1),
    }


@st.cache_data(show_spinner="Predicting all 72 group-stage fixtures …")
def compute_wc2026_predictions(rev: str = ""):
    """Run the trained classifier + goals model over every 2026 group fixture.

    Returns (rows, model_name). Cached so changing UI filters doesn't recompute;
    `rev` (from `_data_rev()`) busts the cache when the model/data changes.
    """
    from src.models.predictor import predict_match
    from src.models.goals_model import predict_fixture
    from src.data.wc2026 import FIXTURES, elo_name, host_advantage
    from src.data.reputation import adjusted_elo

    model, model_name = load_trained_model()
    elos = load_current_elos()
    if model is None:
        return [], None

    rows = []
    for fx in FIXTURES:
        home, away = fx["home"], fx["away"]
        home_elo = adjusted_elo(home, elos.get(elo_name(home), 1500))
        away_elo = adjusted_elo(away, elos.get(elo_name(away), 1500))
        neutral = not host_advantage(home, away)

        outcome = predict_match(
            model, home, away, home_elo, away_elo,
            _elo_stats(home_elo), _elo_stats(away_elo),
            neutral=neutral, is_world_cup=True, is_knockout=False,
        )
        pred = predict_fixture(
            home, away, home_elo, away_elo,
            neutral=neutral, outcome_probs=outcome,
        )
        pred.update({
            "group": fx["group"], "date": fx["date"], "time": fx.get("time", ""),
            "venue": fx.get("venue", ""), "home_elo": home_elo, "away_elo": away_elo,
        })
        rows.append(pred)
    return rows, model_name


@st.cache_data(show_spinner="Running 5,000 World Cup 2026 simulations …")
def compute_wc2026_tournament(n_simulations: int = 5000, rev: str = ""):
    """Monte Carlo the real 2026 bracket with reputation-adjusted Elo.

    Returns a DataFrame with group-standings probabilities, Round-of-32
    qualification odds, and knockout stage reach probabilities.
    """
    from src.simulation.simulator import run_wc2026_simulation
    elos = load_current_elos()
    if not elos:
        return None
    return run_wc2026_simulation(elos, n_simulations=n_simulations, seed=0)


# ── sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "⚽ Match Predictor",
    "📊 Team Comparison",
    "🏆 Tournament Simulator",
    "📈 Probability Charts",
    "🔬 Model Evaluation",
    "🧠 Feature Importance",
]

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FIFA_logo.svg/200px-FIFA_logo.svg.png", width=80)
    st.title("FIFA WC Predictor")
    page = st.radio("Navigate", PAGES)
    st.markdown("---")
    st.caption("Built with Python · scikit-learn · XGBoost · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("⚽ FIFA World Cup Prediction Machine")
    st.markdown("""
    > **An end-to-end ML system** that predicts match outcomes, simulates full tournaments,
    > and explains its reasoning — from raw data to interactive dashboard.
    """)

    c1, c2, c3, c4 = st.columns(4)
    feat = load_features()
    sim = load_sim_results()
    mr = load_model_results()

    with c1:
        st.metric("Historical Matches", f"{len(feat):,}" if feat is not None else "–")
    with c2:
        n_teams = len(sim) if sim is not None else 0
        st.metric("Teams Modelled", n_teams or "–")
    with c3:
        st.metric("MC Simulations", "5,000")
    with c4:
        if mr is not None and not mr.empty:
            best_ll = mr["log_loss"].min()
            st.metric("Best Log-Loss", f"{best_ll:.4f}")
        else:
            st.metric("Models Trained", "–")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Pipeline Overview")
        st.markdown("""
        | Phase | Component | Description |
        |-------|-----------|-------------|
        | 1 | **Data Pipeline** | Downloads 50k+ international results; generates synthetic data offline |
        | 2 | **Feature Engineering** | Rolling Elo, form stats, attack/defense strength |
        | 3 | **Model Training** | LR · RF · XGBoost · LightGBM · CatBoost — time-split evaluation |
        | 4 | **Simulator** | Monte Carlo group + knockout; 5k runs → win probabilities |
        | 5 | **Dashboard** | Interactive Streamlit app with SHAP explainability |
        """)

    with col_b:
        st.subheader("Quick Start")
        st.code("""
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Launch dashboard
streamlit run app/dashboard.py
        """, language="bash")

    if sim is not None:
        st.markdown("---")
        st.subheader("Top 5 Tournament Favourites")
        top5 = sim.head(5)[["team", "p_winner", "p_final", "p_sf"]]
        top5.columns = ["Team", "Win %", "Final %", "Semi-Final %"]
        for col in ["Win %", "Final %", "Semi-Final %"]:
            top5[col] = (top5[col] * 100).round(1)
        st.dataframe(top5, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Match Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("⚽ Match Predictor")
    model, model_name = load_trained_model()
    elos = load_current_elos()

    if model is None:
        st.warning("No trained model found. Run `python main.py` first.")
        st.stop()

    tab_custom, tab_schedule = st.tabs(["🔮 Custom Match", "📅 FIFA 2026 Schedule"])

    # ── Tab 1: build-your-own match ────────────────────────────────────────────
    with tab_custom:
        feat_df = load_features()
        all_teams = sorted(set(
            list(elos.keys()) +
            (list(feat_df["home_team"].unique()) if feat_df is not None else [])
        ))

        st.info(f"Model: **{model_name}** · Elo ratings from training data")

        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home / Team A", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
        with col2:
            away_options = [t for t in all_teams if t != home_team]
            away_team = st.selectbox("Away / Team B", away_options, index=away_options.index("Germany") if "Germany" in away_options else 0)

        col3, col4, col5 = st.columns(3)
        with col3:
            neutral = st.checkbox("Neutral Venue", value=True)
        with col4:
            is_wc = st.checkbox("World Cup Match", value=True)
        with col5:
            is_ko = st.checkbox("Knockout Stage", value=False)

        from src.data.reputation import adjusted_elo, REPUTATION_ELO
        home_elo = adjusted_elo(home_team, elos.get(home_team, 1500))
        away_elo = adjusted_elo(away_team, elos.get(away_team, 1500))

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            from src.models.predictor import predict_match, explain_prediction
            from src.models.goals_model import predict_fixture

            home_stats = _elo_stats(home_elo)
            away_stats = _elo_stats(away_elo)

            pred = predict_match(
                model, home_team, away_team,
                home_elo, away_elo,
                home_stats, away_stats,
                neutral=neutral, is_world_cup=is_wc, is_knockout=is_ko,
            )
            goals = predict_fixture(
                home_team, away_team, home_elo, away_elo,
                neutral=neutral, outcome_probs=pred,
            )

            explanation = explain_prediction(pred, home_elo, away_elo, home_stats, away_stats)

            # Display probabilities
            st.markdown("---")
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric(f"🏆 {home_team} Win", f"{pred['home_win_prob']*100:.1f}%")
            with p2:
                st.metric("🤝 Draw", f"{pred['draw_prob']*100:.1f}%")
            with p3:
                st.metric(f"🏆 {away_team} Win", f"{pred['away_win_prob']*100:.1f}%")

            # Predicted scoreline + goals
            sh, sa = goals["likely_score"]
            g1, g2 = st.columns(2)
            with g1:
                st.metric("⚽ Predicted Score", f"{home_team} {sh} – {sa} {away_team}")
            with g2:
                st.metric("Σ Expected Goals", f"{goals['exp_total_goals']:.1f}")

            # Gauge chart
            labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
            values = [pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]]
            colors = ["#1f77b4", "#aec7e8", "#d62728"]

            fig = go.Figure(go.Bar(
                x=labels, y=[v * 100 for v in values],
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                title="Match Outcome Probabilities",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Predicted scorers
            st.markdown("---")
            st.subheader("🥅 Predicted Goalscorers")
            sc1, sc2 = st.columns(2)
            for col, team, scorers, xg in [
                (sc1, home_team, goals["home_scorers"], goals["home_xg"]),
                (sc2, away_team, goals["away_scorers"], goals["away_xg"]),
            ]:
                with col:
                    st.markdown(f"**{team}** · {xg:.2f} xG")
                    if scorers:
                        sdf = pd.DataFrame(scorers)
                        sdf["prob_score"] = (sdf["prob_score"] * 100).round(0).astype(int).astype(str) + "%"
                        sdf.columns = ["Player", "xG", "Chance to score"]
                        st.dataframe(sdf, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No scorer data for this team.")

            # Scoreline probabilities & match markets
            st.markdown("---")
            st.subheader("📐 Scoreline Probabilities & Match Markets")
            from src.models.goals_model import score_matrix
            mk = goals["markets"]
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Any draw", f"{mk['p_draw']*100:.0f}%")
            mm2.metric("Over 2.5 goals", f"{mk['p_over25']*100:.0f}%")
            mm3.metric("Over 3.5 goals", f"{mk['p_over35']*100:.0f}%")
            mm4.metric("Both teams score", f"{mk['p_btts']*100:.0f}%")

            ms1, ms2 = st.columns([2, 3])
            with ms1:
                st.markdown("**Most likely scorelines**")
                tsdf = pd.DataFrame(mk["top_scores"])
                tsdf["prob"] = (tsdf["prob"] * 100).round(1).astype(str) + "%"
                tsdf.columns = [f"{home_team} – {away_team}", "Probability"]
                st.dataframe(tsdf, use_container_width=True, hide_index=True)
            with ms2:
                mx = 6
                grid = score_matrix(goals["home_xg"], goals["away_xg"], max_goals=mx)
                z = [[grid[(h, a)] * 100 for a in range(mx + 1)] for h in range(mx + 1)]
                fig_h = go.Figure(go.Heatmap(
                    z=z,
                    x=[str(a) for a in range(mx + 1)],
                    y=[str(h) for h in range(mx + 1)],
                    colorscale="Blues",
                    text=[[f"{v:.1f}" for v in row] for row in z],
                    texttemplate="%{text}",
                    showscale=False,
                ))
                fig_h.update_layout(
                    title="Scoreline probability (%)",
                    xaxis_title=f"{away_team} goals",
                    yaxis_title=f"{home_team} goals",
                    template="plotly_dark", height=340,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_h, use_container_width=True)
            st.caption("Full Dixon-Coles adjusted Poisson distribution — every scoreline (incl. high-scoring and draws) carries a probability; the headline score is just the single most likely one.")

            # Explanation
            st.markdown("---")
            st.subheader("🧠 Model Explanation")
            st.info(explanation)

            # Elo comparison (blended form + reputation where available)
            blended = " (form + reputation)" if (home_team in REPUTATION_ELO or away_team in REPUTATION_ELO) else ""
            st.markdown(f"**Ratings{blended}** — {home_team}: `{home_elo:.0f}` | {away_team}: `{away_elo:.0f}` | Difference: `{home_elo - away_elo:+.0f}`")

    # ── Tab 2: real FIFA 2026 schedule with predictions ────────────────────────
    with tab_schedule:
        from src.data.wc2026 import GROUPS

        st.markdown(
            "Real **2026 World Cup** group-stage schedule (48 teams · 12 groups · 72 matches). "
            "Each fixture shows the model's **W / D / L** call, a **predicted scoreline** and "
            "**expected goals** (Elo-driven Poisson), plus the **likely scorers** for each side."
        )

        rows, sched_model = compute_wc2026_predictions(_data_rev())
        if not rows:
            st.warning("No trained model found. Run `python main.py` first.")
            st.stop()

        sched_df = pd.DataFrame(rows)

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            grp = st.selectbox("Group", ["All"] + list(GROUPS.keys()))
        with fc2:
            dates = ["All"] + sorted(sched_df["date"].unique())
            day = st.selectbox("Date", dates)

        view = sched_df
        if grp != "All":
            view = view[view["group"] == grp]
        if day != "All":
            view = view[view["date"] == day]

        st.caption(f"Model: **{sched_model}** · {len(view)} of {len(sched_df)} fixtures shown")

        for _, r in view.iterrows():
            home, away = r["home"], r["away"]
            sh, sa = r["likely_score"]
            probs = {home: r["p_home"], "Draw": r["p_draw"], away: r["p_away"]}
            top = max(probs, key=probs.get)
            badge = "🤔" if r.get("tossup") else {home: "🟦", "Draw": "⬜", away: "🟥"}[top]

            header = (
                f"**Group {r['group']}** · {r['date']}"
                + (f" {r['time']}" if r["time"] else "")
                + f"  —  {home} vs {away}"
            )
            with st.expander(header):
                if r["venue"]:
                    st.caption(f"📍 {r['venue']}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{home} win", f"{r['p_home']*100:.0f}%")
                m2.metric("Draw", f"{r['p_draw']*100:.0f}%")
                m3.metric(f"{away} win", f"{r['p_away']*100:.0f}%")
                m4.metric("Predicted score", f"{sh}–{sa}")

                st.markdown(
                    f"{badge} **Prediction:** {r['verdict']} · "
                    f"Σ expected goals **{r['exp_total_goals']:.1f}** "
                    f"({home} {r['home_xg']:.1f} xG, {away} {r['away_xg']:.1f} xG)"
                )

                mk = r.get("markets")
                if mk:
                    tops = " · ".join(f"{t['score']} ({t['prob']*100:.0f}%)" for t in mk["top_scores"][:5])
                    st.markdown(
                        f"**Likely scorelines:** {tops}  \n"
                        f"**Markets:** draw {mk['p_draw']*100:.0f}% · "
                        f"over 2.5 {mk['p_over25']*100:.0f}% · "
                        f"over 3.5 {mk['p_over35']*100:.0f}% · "
                        f"both score {mk['p_btts']*100:.0f}%"
                    )

                s1, s2 = st.columns(2)
                for col, team, scorers in [
                    (s1, home, r["home_scorers"]),
                    (s2, away, r["away_scorers"]),
                ]:
                    with col:
                        st.markdown(f"**{team} — likely scorers**")
                        if scorers:
                            for sc in scorers:
                                st.write(
                                    f"• {sc['player']} — {sc['prob_score']*100:.0f}% "
                                    f"(⌀ {sc['exp_goals']:.2f})"
                                )
                        else:
                            st.caption("No scorer data.")


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Team Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("📊 Team Comparison")

    feat_df = load_features()
    elos = load_current_elos()

    if feat_df is None:
        st.warning("Run the pipeline first to generate feature data.")
        st.stop()

    all_teams = sorted(feat_df["home_team"].unique())
    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox("Team 1", all_teams, index=0)
    with col2:
        t2_opts = [t for t in all_teams if t != t1]
        t2 = st.selectbox("Team 2", t2_opts, index=0)

    def team_recent_stats(df, team, n=20):
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        recent = df[mask].tail(n)
        wins = draws = losses = gf = ga = 0
        for _, r in recent.iterrows():
            if r["home_team"] == team:
                gf += r.get("home_gf_avg_5", 1.3)
                ga += r.get("home_ga_avg_5", 1.0)
                wins += r.get("home_win_rate_5", 0.5)
                draws += r.get("home_draw_rate_5", 0.25)
            else:
                gf += r.get("away_gf_avg_5", 1.3)
                ga += r.get("away_ga_avg_5", 1.0)
                wins += r.get("away_win_rate_5", 0.5)
                draws += r.get("away_draw_rate_5", 0.25)
        n = max(len(recent), 1)
        return {
            "Win Rate (5)": wins / n,
            "Draw Rate (5)": draws / n,
            "Avg Goals For": gf / n,
            "Avg Goals Against": ga / n,
            "Elo": elos.get(team, 1500),
        }

    s1 = team_recent_stats(feat_df, t1)
    s2 = team_recent_stats(feat_df, t2)

    # Radar chart
    categories = list(s1.keys())
    vals1 = list(s1.values())
    vals2 = list(s2.values())

    # Normalise for radar
    max_vals = [max(v1, v2, 0.01) for v1, v2 in zip(vals1, vals2)]
    n1 = [v / m for v, m in zip(vals1, max_vals)]
    n2 = [v / m for v, m in zip(vals2, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=n1 + [n1[0]], theta=categories + [categories[0]],
                                   fill="toself", name=t1))
    fig.add_trace(go.Scatterpolar(r=n2 + [n2[0]], theta=categories + [categories[0]],
                                   fill="toself", name=t2))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"{t1} vs {t2} — Radar Comparison",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw stats table
    stats_df = pd.DataFrame({"Metric": categories, t1: vals1, t2: vals2})
    stats_df[t1] = stats_df[t1].round(3)
    stats_df[t2] = stats_df[t2].round(3)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Tournament Simulator
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("🏆 Tournament Simulator")

    sim = compute_wc2026_tournament(rev=_data_rev())
    if sim is None or sim.empty:
        st.warning("No Elo ratings found. Run `python main.py` first.")
        st.stop()

    from src.data.wc2026 import GROUPS

    st.success(
        f"Based on 5,000 Monte Carlo runs of the **real 2026 bracket** "
        f"({len(sim)} teams · 12 groups) using reputation-adjusted Elo. "
        f"Round of 32 = top 2 per group + 8 best third-placed teams."
    )

    sim_tabs = st.tabs(["🏆 Title race", "📋 Group standings", "🎟️ Road to Round of 32"])

    # ── Title race ─────────────────────────────────────────────────────────────
    with sim_tabs[0]:
        fig = px.bar(
            sim.head(16), x="team", y="p_winner",
            color="p_winner", color_continuous_scale="Viridis",
            title="Tournament Winner Probabilities (Top 16)",
            labels={"p_winner": "Win Probability", "team": "Team"},
            text=sim.head(16)["p_winner"].mul(100).round(1).astype(str) + "%",
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        stages = ["p_ro32", "p_r16", "p_qf", "p_sf", "p_final", "p_winner"]
        stage_labels = ["Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Final", "Winner"]

        top_teams = sim.head(20)
        heat_data = top_teams[stages].values * 100
        fig2 = go.Figure(go.Heatmap(
            z=heat_data, x=stage_labels, y=top_teams["team"].tolist(),
            colorscale="YlOrRd", text=np.round(heat_data, 1),
            texttemplate="%{text}%", showscale=True,
        ))
        fig2.update_layout(
            title="Stage-by-Stage Probabilities (Top 20 Teams)",
            template="plotly_dark", height=600, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full Results Table")
        display = sim.copy()
        for col in stages:
            display[col] = (display[col] * 100).round(1)
        display = display[["team", "group", "elo"] + stages]
        display.columns = ["Team", "Group", "Elo"] + stage_labels
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Group standings ────────────────────────────────────────────────────────
    with sim_tabs[1]:
        st.markdown(
            "Probability each team finishes **1st / 2nd / 3rd / 4th** in its group, "
            "and the chance of reaching the **Round of 32**."
        )
        grp = st.selectbox("Group", list(GROUPS.keys()), key="sim_group")
        gdf = sim[sim["group"] == grp].copy()
        # order by most-likely finishing position
        gdf["rank_key"] = gdf["p_pos1"] * 4 + gdf["p_pos2"] * 3 + gdf["p_pos3"] * 2 + gdf["p_pos4"]
        gdf = gdf.sort_values("rank_key", ascending=False)

        pos_cols = ["p_pos1", "p_pos2", "p_pos3", "p_pos4"]
        pos_labels = ["1st", "2nd", "3rd", "4th"]
        pos_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

        fig_g = go.Figure()
        for col, lab, c in zip(pos_cols, pos_labels, pos_colors):
            fig_g.add_trace(go.Bar(
                name=lab, x=gdf["team"], y=gdf[col] * 100, marker_color=c,
                text=[f"{v*100:.0f}%" if v >= 0.05 else "" for v in gdf[col]],
                textposition="inside",
            ))
        fig_g.update_layout(
            barmode="stack", template="plotly_dark", height=420,
            title=f"Group {grp} — Finishing-Position Probabilities",
            yaxis_title="Probability (%)", yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_g, use_container_width=True)

        tbl = gdf[["team", "elo"] + pos_cols + ["p_ro32"]].copy()
        for col in pos_cols + ["p_ro32"]:
            tbl[col] = (tbl[col] * 100).round(1)
        tbl.columns = ["Team", "Elo", "1st %", "2nd %", "3rd %", "4th %", "Reach R32 %"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.caption("Top 2 in each group qualify automatically; 3rd-placed teams compete for the 8 best-third spots.")

    # ── Road to Round of 32 ──────────────────────────────────────────────────────
    with sim_tabs[2]:
        st.markdown(
            "Probability of reaching the **Round of 32** (48 → 32). The 24 group "
            "winners/runners-up plus the 8 best third-placed teams advance."
        )
        ro32 = sim.sort_values("p_ro32", ascending=False).copy()

        # Likely qualifiers = highest 32 by qualification probability
        cutoff = ro32.iloc[31]["p_ro32"] if len(ro32) >= 32 else 0
        ro32["status"] = ["✅ Likely" if i < 32 else "❌ Bubble/Out" for i in range(len(ro32))]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Most likely to qualify (projected top 32)**")
            qual = ro32.head(32)[["team", "group", "p_ro32"]].copy()
            qual["p_ro32"] = (qual["p_ro32"] * 100).round(1)
            qual.columns = ["Team", "Group", "Qualify %"]
            st.dataframe(qual, use_container_width=True, hide_index=True, height=420)
        with c2:
            st.markdown("**On the bubble / unlikely (ranks 33–48)**")
            out = ro32.iloc[32:][["team", "group", "p_ro32"]].copy()
            out["p_ro32"] = (out["p_ro32"] * 100).round(1)
            out.columns = ["Team", "Group", "Qualify %"]
            st.dataframe(out, use_container_width=True, hide_index=True, height=420)

        fig_r = px.bar(
            ro32.head(24), x="p_ro32", y="team", orientation="h",
            color="p_ro32", color_continuous_scale="Greens",
            title="Round-of-32 Qualification Probability (Top 24)",
            labels={"p_ro32": "Qualify Probability", "team": "Team"},
        )
        fig_r.update_layout(template="plotly_dark", height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — Probability Charts
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("📈 Probability Charts")
    sim = load_sim_results()
    feat_df = load_features()

    if sim is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.treemap(
                sim, path=["group", "team"], values="p_winner",
                color="p_winner", color_continuous_scale="Blues",
                title="Winner Probability Treemap",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                sim, x="elo", y="p_winner",
                text="team", color="group",
                size="p_winner", size_max=30,
                title="Elo vs Win Probability",
                labels={"elo": "Elo Rating", "p_winner": "Win Probability"},
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

    if feat_df is not None:
        st.markdown("---")
        st.subheader("Historical Match Outcome Distribution")
        outcome_counts = feat_df["target"].value_counts().sort_index()
        outcome_counts.index = ["Away Win", "Draw", "Home Win"]
        fig3 = px.pie(values=outcome_counts.values, names=outcome_counts.index,
                      title="Match Outcome Distribution (Training Data)",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 6 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("🔬 Model Evaluation")
    mr = load_model_results()

    if mr is None:
        st.warning("No model results found. Run `python main.py` first.")
        st.stop()

    mr_display = mr.round(4).reset_index()
    mr_display.columns = ["Model", "Accuracy", "Log-Loss", "Brier Score"]

    # Highlight best in each metric
    st.subheader("Comparative Metrics")
    st.dataframe(mr_display, use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig = go.Figure()
    metrics = ["Accuracy", "Log-Loss", "Brier Score"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=mr_display["Model"],
            y=mr_display[metric],
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        template="plotly_dark",
        yaxis_title="Score",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Metric Guide**
    - **Accuracy** — fraction of correct outcome predictions (higher is better)
    - **Log-Loss** — penalises confident wrong predictions (lower is better)
    - **Brier Score** — mean squared error of probabilities (lower is better)

    > The best model for probabilistic predictions is typically the one with lowest Log-Loss.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Page 7 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("🧠 Feature Importance")

    model, model_name = load_trained_model()
    if model is None:
        st.warning("No trained model found. Run `python main.py` first.")
        st.stop()

    from src.models.explainer import get_feature_importance
    importance = get_feature_importance(model, model_name)

    if importance.empty:
        st.warning("Feature importance not available for this model type.")
        st.stop()

    top_n = st.slider("Show top N features", 5, len(importance), 20)
    top = importance.head(top_n).sort_values()

    fig = px.bar(
        x=top.values, y=top.index,
        orientation="h",
        color=top.values,
        color_continuous_scale="Teal",
        title=f"Feature Importance — {model_name}",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(template="plotly_dark", height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Descriptions")
    st.markdown("""
    | Feature | Description |
    |---------|-------------|
    | `elo_diff` | Pre-match Elo rating difference (home − away) |
    | `home_elo` / `away_elo` | Absolute Elo ratings |
    | `conf_elo_diff` | Confederation strength proxy difference |
    | `home_win_rate_5` | Fraction of last 5 matches won (home team) |
    | `gd_diff_5` | Goal difference per game differential (last 5) |
    | `home_attack_strength` | Goals scored / league average (last 10) |
    | `away_defense_strength` | Goals conceded / league average (last 10) |
    | `attack_diff` | Home attack strength − away attack strength |
    | `neutral_venue` | 1 if match is on neutral ground |
    | `is_world_cup` | 1 if this is a World Cup match |
    | `is_knockout` | 1 if this is a knockout-stage match |
    """)

    # SHAP (optional)
    feat_df = load_features()
    if feat_df is not None:
        st.markdown("---")
        if st.checkbox("Run SHAP analysis (may take ~10 seconds)"):
            from src.models.explainer import compute_shap_values
            from src.features.builder import FEATURE_COLS
            X = feat_df[FEATURE_COLS].fillna(0).values[:500]
            shap_vals, _ = compute_shap_values(model, X, model_name)
            if shap_vals is not None:
                shap_path = Path(f"reports/shap_summary_{model_name}.png")
                if shap_path.exists():
                    st.image(str(shap_path), caption="SHAP Summary Plot")
                else:
                    st.info("SHAP values computed. Re-run `python main.py` to generate the plot.")
            else:
                st.warning("SHAP analysis not available for this model.")

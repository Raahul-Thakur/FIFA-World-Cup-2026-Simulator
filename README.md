# ⚽ FIFA World Cup Prediction Machine

An end-to-end machine learning system that predicts FIFA World Cup match outcomes,
simulates full tournament brackets with Monte Carlo, and surfaces results through
an interactive Streamlit dashboard.

Built as a portfolio project showcasing **data engineering**, **feature engineering**,
**ML model comparison**, **probabilistic simulation**, and **ML explainability**.

---

## Why This Project Matters

Predicting football outcomes is a genuinely hard probabilistic problem.  Teams play
only ~10–15 competitive matches per year, results are noisy, and upsets are frequent.
This project tackles that head-on with:

- **Temporal Elo ratings** that track team strength through every match in history
- **Rolling form features** that capture momentum without leaking future information
- **Multiple calibrated classifiers** compared head-to-head on a held-out time slice
- **Monte Carlo simulation** that propagates uncertainty through an entire tournament
- **SHAP explainability** that explains individual predictions in plain English

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data | `pandas`, `numpy`, `requests` |
| Features | Custom rolling Elo + form engine |
| Models | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Explainability | `shap`, `matplotlib` |
| Simulation | NumPy Monte Carlo (10,000 runs) |
| Dashboard | `streamlit`, `plotly` |
| Persistence | `joblib` |

---

## Dataset Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| International match results (1872–present) | [martj42/international_results](https://github.com/martj42/international_results) | ~50,000 matches, free, no API key |
| Penalty shootout results | Same repository | Used to resolve simulated knockout draws |
| Confederation / team metadata | Hardcoded from FIFA | Easy to extend with live FIFA rankings |

The downloader falls back to **synthetic data** when the network is unavailable,
so the entire pipeline runs offline for development and CI.

---

## ML Methodology

### Time-Based Train/Test Split
All models are trained on matches **before 2018** and evaluated on matches **2018 onward**.
This mirrors real deployment: you never know future results when training.

### Features (40 total)

| Category | Features |
|----------|----------|
| Elo | `elo_diff`, `home_elo`, `away_elo`, `conf_elo_diff` |
| Short-form (5 matches) | win rate, draw rate, goals for/against, goal diff |
| Long-form (10 matches) | same as above |
| Derived form | win rate diff, gd diff (home − away) |
| Strength | attack strength, defense strength, attack diff, defense diff |
| Context | neutral venue, host nation, is World Cup, is knockout |

### Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline; fast, interpretable |
| Random Forest | Handles non-linear interactions well |
| XGBoost | Gradient boosted trees; typically best log-loss |
| LightGBM | Fast alternative to XGBoost |
| CatBoost | Good with categorical features |

### Evaluation Metrics

- **Accuracy** — fraction of correct outcome predictions
- **Log-Loss** — rewards calibrated probabilities, penalises confident errors
- **Brier Score** — MSE of predicted probabilities

---

## Simulation Methodology

1. **Group stage** — round-robin; Elo win probabilities drive match outcomes;
   Poisson-distributed scorelines track group statistics (GF, GD)
2. **Tiebreakers** — points → goal difference → goals scored → random (coin flip)
3. **Knockout stage** — single-elimination; drawn matches resolved 50/50 by penalty
4. **10,000 independent runs** — each team's tournament path is tracked;
   final probability = fraction of runs where team reached that stage

---

## Project Structure

```
worldcup-predictor/
├── data/
│   ├── raw/                    # downloaded CSV files (git-ignored)
│   └── processed/              # cleaned features, simulation results
├── models_saved/               # serialised .joblib model files
├── reports/                    # evaluation plots, SHAP charts
├── notebooks/                  # exploratory analysis (add your own)
├── src/
│   ├── data/
│   │   ├── downloader.py       # fetch real data or generate synthetic
│   │   └── preprocessor.py     # clean, normalise, derive outcome columns
│   ├── features/
│   │   ├── elo.py              # rolling Elo computation
│   │   ├── form.py             # rolling form stats per team
│   │   └── builder.py          # assemble final feature matrix + FEATURE_COLS list
│   ├── models/
│   │   ├── trainer.py          # train / evaluate / save all models
│   │   ├── predictor.py        # single-match prediction + NL explanation
│   │   └── explainer.py        # feature importance + SHAP
│   ├── simulation/
│   │   └── simulator.py        # Monte Carlo tournament simulator
│   └── utils/
│       ├── config.py           # paths, constants, Elo parameters
│       └── logger.py           # consistent logging across modules
├── app/
│   └── dashboard.py            # 7-page Streamlit dashboard
├── main.py                     # pipeline orchestrator (CLI)
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone / copy the project

```bash
cd worldcup-predictor
```

This repository includes trained model artifacts in `models_saved/` using
Git LFS. After cloning from GitHub, install Git LFS and run:

```bash
git lfs pull
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
- Download ~50k historical match results from GitHub
- Compute rolling Elo ratings and form features
- Train and compare 4–5 models
- Run 10,000 Monte Carlo tournament simulations
- Save models and results to disk

**Faster test run** (skips heavy form computation, uses 1k simulations):

```bash
python main.py --fast
```

**Re-use cached features** (skip re-computing form, re-train models only):

```bash
python main.py --skip-features
```

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Screenshots

| Page | Description |
|------|-------------|
| Home | Pipeline overview + top favourites |
| Match Predictor | Select any two teams → win/draw/loss probabilities + explanation |
| Team Comparison | Radar chart comparing two teams across 5 dimensions |
| Tournament Simulator | Heatmap of stage-by-stage probabilities for all 32 teams |
| Probability Charts | Treemap + Elo vs probability scatter |
| Model Evaluation | Bar chart comparing accuracy, log-loss, Brier score |
| Feature Importance | Interactive bar chart + optional SHAP summary |

*(Add screenshots here after running the app)*

---

## Future Improvements

- **Player-level data** — squad value (Transfermarkt), injury lists, key player absence
- **Betting odds** — market-implied probabilities as a powerful baseline feature
- **Live rankings** — pull current FIFA rankings via API for real-time updates
- **Neural network** — PyTorch FFNN or a transformer on match sequences
- **Expected goals (xG)** — more predictive than raw goal counts
- **Venue/climate effects** — altitude, temperature, travel distance
- **Tournament-specific calibration** — separate model for WC knockout pressure
- **REST API** — wrap the predictor in FastAPI for external consumption
- **Docker** — containerise for one-command deployment

---

## License

MIT — free to use, modify, and distribute.

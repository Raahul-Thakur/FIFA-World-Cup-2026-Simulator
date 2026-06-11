"""Central configuration for paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models_saved"
REPORTS_DIR = ROOT / "reports"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Elo system constants
ELO_K_FACTOR = 32        # base K-factor; adjusted by match importance
ELO_HOME_ADVANTAGE = 100 # Elo points added for home advantage
INITIAL_ELO = 1500       # starting Elo for teams with no history

# Feature engineering windows
FORM_WINDOW_SHORT = 5    # last N matches for short-form features
FORM_WINDOW_LONG = 10    # last N matches for long-form features

# Simulation
N_SIMULATIONS = 10_000

# Confederation Elo strength proxy (rough historical estimate)
CONFEDERATION_STRENGTH = {
    "UEFA": 1600,
    "CONMEBOL": 1620,
    "CONCACAF": 1480,
    "CAF": 1460,
    "AFC": 1450,
    "OFC": 1380,
}

WORLD_CUP_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970,
    1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006,
    2010, 2014, 2018, 2022,
]

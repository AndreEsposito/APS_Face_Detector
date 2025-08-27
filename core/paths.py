from __future__ import annotations
from pathlib import Path

# Pastas principais
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CAPTURES_DIR = ROOT / "captures"

# Arquivos
LBF_MODEL_PATH_DEFAULT = MODELS_DIR / "lbfmodel.yaml"
LBPH_MODEL_PATH = MODELS_DIR / "lbph.yml"
SQLITE_PATH = DATA_DIR / "db.sqlite"
ACCESS_LOG_CSV = REPORTS_DIR / "access_log.csv"

def ensure_dirs():
    for p in [DATA_DIR, SAMPLES_DIR, MODELS_DIR, REPORTS_DIR, CAPTURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)

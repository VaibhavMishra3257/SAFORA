from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRENDS_PATH = DATA_RAW / "trends.csv"
ANILIST_PATH = DATA_RAW / "anime_metadata.csv"
REGIONAL_PATH = DATA_RAW / "regional_trends.csv"
MODEL_PATH = MODELS / "demand_model.joblib"
METRICS_PATH = REPORTS / "model_metrics.json"

for directory in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

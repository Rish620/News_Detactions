from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "news.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

LABEL_NAMES = {
    0: "FAKE",
    1: "REAL",
}


import pandas as pd

from .config import DATA_PATH


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load and clean the labeled training data."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset=["text"])


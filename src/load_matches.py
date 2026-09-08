"""Read the untouched CSV into a pandas DataFrame."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data/raw/2022-23.csv"


def load_matches(path: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

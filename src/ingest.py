import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def ingest_data(file_name_or_path="Customer-Churn.csv"):
    path = Path(file_name_or_path)
    if not path.exists() and not path.is_absolute():
        raw_path = BASE_DIR / "data" / "raw" / file_name_or_path
        if raw_path.exists():
            path = raw_path
    df = pd.read_csv(path)
    return df
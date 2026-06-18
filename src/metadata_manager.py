from pathlib import Path
from datetime import datetime
from src.utils import save_json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_metadata(df, dataset_name="Dataset", target_col=None):
    metadata = {}

    # 🔹 Basic info
    metadata['dataset_name'] = dataset_name
    metadata['num_rows'] = df.shape[0]
    metadata['num_columns'] = df.shape[1]

    # 🔹 Columns
    metadata['columns'] = list(df.columns)

    # 🔹 Data types
    metadata['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 🔹 Target column auto-detection
    if target_col is None:
        target_col = "Churn" if "Churn" in df.columns else df.columns[-1]
    metadata['target_column'] = target_col

    # 🔹 Owner
    metadata['owner'] = "Rohan Mahendra"

    # 🔹 Timestamp
    metadata['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save metadata
    save_json(metadata, METADATA_DIR / "dataset_catalog.json")

    return metadata
from pathlib import Path
from datetime import datetime
from src.utils import save_json, load_json


BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)


def log_step(step_name, description=""):
    log_file = METADATA_DIR / "lineage_log.json"

    lineage = load_json(log_file, default=[])

    entry = {
        "step": step_name,
        "description": description,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    lineage.append(entry)

    save_json(lineage, log_file)

    return entry
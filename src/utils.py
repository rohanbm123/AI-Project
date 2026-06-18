import json
from pathlib import Path

def save_json(data, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return data

def load_json(file_path, default=None):
    file_path = Path(file_path)
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default
import json
from pathlib import Path


def write_json(path, value):
    """Save readable results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def new_run_folder(path):
    """Refuse to mix a new training run with an old one."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output folder is not empty: {path}. Choose a new output path.")
    path.mkdir(parents=True, exist_ok=True)
    return path

"""Shared paths, input validation, and plotting setup for Part 01 scripts."""
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def prepare_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_weather(path: Path, required: tuple[str, ...] = ("date", "temp_mean", "rainfall")):
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Required weather data not found: {path}. Run scripts/01_data_collection.py first.")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError(f"No valid dated weather rows found in {path}")
    return frame.reset_index(drop=True)

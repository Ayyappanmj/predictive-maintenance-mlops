"""Data ingestion: config loading, dataset download and raw loading."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_dataset(cfg: dict, force: bool = False, timeout: int = 60) -> Path:
    """Download the AI4I 2020 zip from UCI and save the CSV to cfg['data']['raw_path']."""
    out = ROOT / cfg["data"]["raw_path"]
    if out.exists() and not force:
        logger.info("Dataset already present at %s", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["data"]["uci_zip_url"]
    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError(f"No CSV found in downloaded archive: {zf.namelist()}")
        member = next((n for n in csvs if "ai4i" in n.lower()), csvs[0])
        out.write_bytes(zf.read(member))
    logger.info("Saved dataset to %s", out)
    return out


def load_raw(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["raw_path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path)

"""Download and validate the AI4I 2020 dataset.

Usage: python scripts/download_data.py [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingestion import download_dataset, load_config, load_raw  # noqa: E402
from src.data_validation import validate_dataframe  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args()
    cfg = load_config()
    path = download_dataset(cfg, force=args.force)
    rep = validate_dataframe(load_raw(cfg), cfg)
    print(f"Saved: {path}")
    print(f"Validation ok: {rep.ok}")
    print(f"Stats: {rep.stats}")
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR: {e}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

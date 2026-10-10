"""Run a drift check of a CSV (default: the project dataset) against the approved model's
reference data. Informational: exits 0 unless --fail-on-alert is given.
With the static AI4I dataset this normally shows no drift; it demonstrates the pipeline.
Usage: python scripts/run_drift_check.py [--current data/ai4i2020.csv --sample 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingestion import ROOT, load_config  # noqa: E402
from src.drift_monitor import detect_drift, load_reference, save_report  # noqa: E402
from src.preprocessing import input_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default="data/ai4i2020.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--fail-on-alert", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    model_dir = ROOT / a.model_dir
    if not (model_dir / "reference.csv").exists():
        print("No approved model/reference found; skipping drift check (first run).")
        return 0
    meta = json.loads((model_dir / "metadata.json").read_text())
    cur = pd.read_csv(ROOT / a.current)[input_columns(cfg)]
    cur = cur.sample(min(a.sample, len(cur)), random_state=cfg["seed"])
    res = detect_drift(load_reference(model_dir), cur, cfg, meta["model_version"])
    save_report(res, ROOT / a.report_dir)
    print(f"Drift check: alert={res['alert_level']} drifted={res['drifted_features']}")
    return 1 if (a.fail_on_alert and res["alert_level"] == "alert") else 0


if __name__ == "__main__":
    raise SystemExit(main())

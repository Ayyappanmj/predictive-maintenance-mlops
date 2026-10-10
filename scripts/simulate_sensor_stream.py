"""Generate a SIMULATED sensor stream and run drift monitoring on every batch.

Usage: python scripts/simulate_sensor_stream.py [--batches 12 --batch-size 200
       --drift-start 6 --strength 2.0 --seed 42 --model-dir models --out-dir reports]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingestion import ROOT, load_config  # noqa: E402
from src.drift_monitor import detect_drift, load_reference, save_report  # noqa: E402
from src.preprocessing import input_columns  # noqa: E402
from src.simulation import simulate_stream  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--drift-start", type=int, default=6)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    cfg = load_config()
    model_dir = Path(a.model_dir) if Path(a.model_dir).is_absolute() else ROOT / a.model_dir
    out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_reference(model_dir)
    version = json.loads((model_dir / "metadata.json").read_text())["model_version"]

    stream = simulate_stream(ref, cfg, a.batches, a.batch_size, a.drift_start, a.strength, a.seed)
    stream.to_csv(out_dir / "simulated_stream.csv", index=False)
    print("SIMULATED stream (timestamps are simulated, not measurements)")
    print(f"{'batch':>5} {'shift injected':>14} {'drifted':>8} {'share':>6} alert")
    summary = []
    for b, g in stream.groupby("batch_id"):
        r = detect_drift(ref, g[input_columns(cfg)], cfg, version)
        summary.append(
            {
                "batch": int(b),
                "injected_shift": bool(g["injected_shift"].iloc[0]),
                "n_drifted": r["n_drifted"],
                "drift_share": r["drift_share"],
                "alert_level": r["alert_level"],
                "drifted_features": r["drifted_features"],
            }
        )
        print(
            f"{b:>5} {str(summary[-1]['injected_shift']):>14} {r['n_drifted']:>8} "
            f"{r['drift_share']:>6.2f} {r['alert_level']}"
        )
        last = r
    (out_dir / "stream_drift_summary.json").write_text(json.dumps(summary, indent=2))
    save_report(last, out_dir)
    print(f"Wrote simulated_stream.csv, stream_drift_summary.json, drift_report.* to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

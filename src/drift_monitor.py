"""Data drift monitoring.

Core detection uses SciPy statistical tests (open-source, lightweight, fully tested here):
  numeric features     -> two-sample Kolmogorov-Smirnov test (+ PSI as extra information)
  categorical features -> chi-square test of homogeneity
An optional Evidently HTML report is available via `evidently_html_report` if installed.

Data drift is not concept drift: a change in input distributions does NOT prove that model
accuracy dropped. Measuring accuracy requires reliable ground-truth failure labels.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data_ingestion import ROOT
from src.data_validation import select_features_target, validate_sensor_values
from src.preprocessing import input_columns

NOTE = (
    "Data drift means input distributions changed versus the reference (training) data. "
    "It does not by itself mean predictions became less accurate (that would be concept "
    "drift, which needs reliable ground-truth labels to measure)."
)


def create_reference(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reference data = model input columns of the TRAINING split (same split as train.py)."""
    from src.train import split_data  # local import avoids a circular import

    X, y = select_features_target(df, cfg)
    X_tr, *_ = split_data(X, y, cfg)
    return X_tr.reset_index(drop=True)


def load_reference(model_dir=None) -> pd.DataFrame:
    d = Path(model_dir or ROOT / "models")
    path = d / "reference.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Re-run `python -m src.train`.")
    return pd.read_csv(path)


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _numeric_feature(name: str, ref: pd.Series, cur: pd.Series, d: dict) -> dict:
    ks = stats.ks_2samp(ref.to_numpy(float), cur.to_numpy(float))
    psi = _psi(ref.to_numpy(float), cur.to_numpy(float), d["psi_bins"])
    return {
        "name": name,
        "type": "numerical",
        "test": "ks_2samp",
        "statistic": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "psi": psi,
        "psi_flag": bool(psi >= d["psi_flag"]),
        "ref_mean": float(ref.mean()),
        "cur_mean": float(cur.mean()),
        "drifted": bool(ks.pvalue < d["alpha"]),
    }


def _categorical_feature(name: str, ref: pd.Series, cur: pd.Series, d: dict) -> dict:
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    table = np.array(
        [
            [int((ref.astype(str) == c).sum()) for c in cats],
            [int((cur.astype(str) == c).sum()) for c in cats],
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        stat, p = 0.0, 1.0
    else:
        stat, p, _, _ = stats.chi2_contingency(table)
    return {
        "name": name,
        "type": "categorical",
        "test": "chi2",
        "statistic": float(stat),
        "p_value": float(p),
        "psi": None,
        "psi_flag": False,
        "ref_mean": None,
        "cur_mean": None,
        "drifted": bool(p < d["alpha"]),
    }


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, cfg: dict, model_version: str | None = None
) -> dict:
    """Compare a current sensor batch with the reference data. Raises ValueError on bad input."""
    d = cfg["drift"]
    cols = input_columns(cfg)
    missing = [c for c in cols if c not in current.columns]
    if missing:
        raise ValueError(f"Missing columns in current batch: {missing}")
    cur = current[cols]
    if len(cur) < d["min_batch_size"]:
        raise ValueError(
            f"Batch has {len(cur)} rows; at least {d['min_batch_size']} are "
            "required for reliable drift tests"
        )
    errors = validate_sensor_values(cur, cfg)
    for c in cfg["data"]["categorical_features"]:
        if cur[c].isna().any():
            errors.append(f"{c}: missing values")
        bad = set(cur[c].dropna().astype(str)) - set(cfg["validation"]["allowed_types"])
        if bad:
            errors.append(f"{c}: unexpected categories {sorted(bad)}")
    if errors:
        raise ValueError("; ".join(errors))

    ref = reference[cols]
    feats = []
    for c in cfg["data"]["numeric_features"]:
        feats.append(_numeric_feature(c, ref[c], cur[c], d))
    for c in cfg["data"]["categorical_features"]:
        feats.append(_categorical_feature(c, ref[c], cur[c], d))

    drifted = [f["name"] for f in feats if f["drifted"]]
    share = len(drifted) / len(feats)
    level = "alert" if share >= d["alert_share"] else "warning" if drifted else "none"
    return {
        "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "n_features": len(feats),
        "n_drifted": len(drifted),
        "drift_share": float(share),
        "dataset_drift": bool(share >= d["alert_share"]),
        "alert_level": level,
        "drifted_features": drifted,
        "features": feats,
        "thresholds": {
            "alpha": d["alpha"],
            "alert_share": d["alert_share"],
            "min_batch_size": d["min_batch_size"],
            "psi_flag": d["psi_flag"],
        },
        "note": NOTE,
    }


def render_html_report(result: dict) -> str:
    def psi_text(f: dict) -> str:
        return "" if f["psi"] is None else format(f["psi"], ".3f")

    rows = "".join(
        f"<tr><td>{html.escape(f['name'])}</td><td>{f['type']}</td><td>{f['test']}</td>"
        f"<td>{f['statistic']:.4f}</td><td>{f['p_value']:.4g}</td>"
        f"<td>{psi_text(f)}</td>"
        f"<td>{'DRIFT' if f['drifted'] else 'ok'}</td></tr>"
        for f in result["features"]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 10px}}th{{background:#eee}}</style></head><body>
<h1>Data drift report</h1>
<p><b>Alert level:</b> {result["alert_level"].upper()} &middot;
{result["n_drifted"]}/{result["n_features"]} features drifted &middot;
model version: {html.escape(str(result["model_version"]))} &middot;
generated: {html.escape(result["monitoring_timestamp"])}</p>
<p>Reference rows: {result["n_reference"]} &middot; current rows: {result["n_current"]}</p>
<table><tr><th>Feature</th><th>Type</th><th>Test</th><th>Statistic</th><th>p-value</th>
<th>PSI</th><th>Status</th></tr>{rows}</table>
<p><i>{html.escape(result["note"])}</i></p></body></html>"""


def save_report(result: dict, report_dir) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    j, h = report_dir / "drift_report.json", report_dir / "drift_report.html"
    j.write_text(json.dumps(result, indent=2))
    h.write_text(render_html_report(result), encoding="utf-8")
    return j, h


def evidently_html_report(reference: pd.DataFrame, current: pd.DataFrame, path) -> Path:
    """OPTIONAL: Evidently's own HTML drift report. Evidently's API differs across versions;
    both the legacy (<0.7) and new (>=0.7) APIs are attempted. Not covered by the test suite."""
    try:
        try:  # new API (>= 0.7)
            from evidently import Report
            from evidently.presets import DataDriftPreset

            snap = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
            snap.save_html(str(path))
        except ImportError:  # legacy API
            from evidently.metric_preset import DataDriftPreset as LegacyPreset
            from evidently.report import Report as LegacyReport

            rep = LegacyReport(metrics=[LegacyPreset()])
            rep.run(reference_data=reference, current_data=current)
            rep.save_html(str(path))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Evidently report failed (version mismatch?): {e}") from e
    return Path(path)

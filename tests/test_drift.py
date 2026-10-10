"""Drift detection and simulation tests (synthetic data; seeds fixed for reproducibility)."""

import json

import pandas as pd

from src.drift_monitor import create_reference, detect_drift, render_html_report
from src.preprocessing import input_columns
from src.simulation import simulate_stream
from tests.helpers import fast_config, make_synthetic_df

CFG = fast_config()
COLS = input_columns(CFG)
REF = create_reference(make_synthetic_df(), CFG)


def _current(seed=11, n=300):
    return make_synthetic_df(n, seed)[COLS]


def test_reference_has_only_input_columns():
    assert list(REF.columns) == COLS and len(REF) < 1500


def test_normal_data_no_dataset_drift():
    r = detect_drift(REF, _current(), CFG, "vtest")
    assert r["alert_level"] != "alert" and not r["dataset_drift"]


def test_shifted_numeric_data_detected():
    cur = _current()
    cur["Torque [Nm]"] += 25
    cur["Tool wear [min]"] += 80
    cur["Air temperature [K]"] += 6
    r = detect_drift(REF, cur, CFG)
    assert r["alert_level"] == "alert" and r["dataset_drift"]
    assert {"Torque [Nm]", "Tool wear [min]", "Air temperature [K]"} <= set(r["drifted_features"])


def test_categorical_drift_detected():
    cur = _current()
    cur["Type"] = "H"
    r = detect_drift(REF, cur, CFG)
    f = next(x for x in r["features"] if x["name"] == "Type")
    assert f["drifted"] and f["type"] == "categorical"


def test_report_contents_and_serialisation():
    r = detect_drift(REF, _current(), CFG, "v123")
    assert r["model_version"] == "v123" and r["monitoring_timestamp"]
    assert r["n_features"] == len(COLS) and "concept drift" in r["note"]
    json.dumps(r)  # must be JSON-serialisable
    page = render_html_report(r)
    assert "Torque [Nm]" in page and "v123" in page


def test_alert_threshold_is_configurable():
    cur = _current()
    cur["Torque [Nm]"] += 25
    strict = json.loads(json.dumps(CFG))
    strict["drift"]["alert_share"] = 0.99
    assert detect_drift(REF, cur, strict)["alert_level"] == "warning"
    lax = json.loads(json.dumps(CFG))
    lax["drift"]["alert_share"] = 0.10
    assert detect_drift(REF, cur, lax)["alert_level"] == "alert"


def test_bad_batches_raise():
    for bad in (_current(n=10), _current().drop(columns=["Torque [Nm]"])):
        try:
            detect_drift(REF, bad, CFG)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    cur = _current()
    cur.loc[0, "Torque [Nm]"] = float("nan")
    try:
        detect_drift(REF, cur, CFG)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "missing" in str(e)


def test_simulation_is_reproducible():
    a = simulate_stream(REF, CFG, 6, 100, 3, seed=7)
    b = simulate_stream(REF, CFG, 6, 100, 3, seed=7)
    c = simulate_stream(REF, CFG, 6, 100, 3, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a[COLS[:5]].equals(c[COLS[:5]])
    assert "simulated_timestamp" in a.columns and a["injected_shift"].sum() == 3 * 100


def test_simulated_stream_drift_appears_after_injection():
    s = simulate_stream(REF, CFG, 10, 200, 5, 2.5, seed=42)
    levels = {}
    for b, g in s.groupby("batch_id"):
        levels[b] = detect_drift(REF, g[COLS], CFG)["alert_level"]
    assert all(levels[b] != "alert" for b in range(0, 5))
    assert all(levels[b] == "alert" for b in range(7, 10))

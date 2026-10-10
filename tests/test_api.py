"""API tests using FastAPI's TestClient and a model trained on synthetic data."""

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from src.train import train_pipeline
from tests.helpers import fast_config, make_synthetic_df

_TMP = tempfile.TemporaryDirectory()
MODEL_DIR = Path(_TMP.name) / "m"
train_pipeline(make_synthetic_df(), fast_config(), MODEL_DIR, Path(_TMP.name) / "r", False)

VALID = {
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
    "product_type": "M",
}


def test_health_with_model():
    with TestClient(create_app(MODEL_DIR)) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_health_without_model_is_degraded_but_200():
    with tempfile.TemporaryDirectory() as empty:
        with TestClient(create_app(empty)) as c:
            r = c.get("/health")
            assert r.status_code == 200 and r.json()["status"] == "degraded"


def test_model_info():
    with TestClient(create_app(MODEL_DIR)) as c:
        body = c.get("/model-info").json()
        assert body["model_version"] and "test_metrics" in body
        assert "not a confirmed" in body["disclaimer"]


def test_predict_response_schema():
    with TestClient(create_app(MODEL_DIR)) as c:
        r = c.post("/predict", json=VALID)
        assert r.status_code == 200
        b = r.json()
        assert 0 <= b["failure_probability"] <= 1
        assert b["risk_category"] in {"low", "medium", "high"}
        assert b["model_version"] and b["prediction_timestamp"]
        assert "not a confirmed" in b["disclaimer"]


def test_predict_rejects_invalid_values():
    with TestClient(create_app(MODEL_DIR)) as c:
        for bad in (
            {**VALID, "torque_nm": -5},
            {**VALID, "product_type": "X"},
            {**VALID, "tool_wear_min": "abc"},
            {**VALID, "extra": 1},
        ):
            r = c.post("/predict", json=bad)
            assert r.status_code == 422
            assert r.json()["error"]["code"] == "validation_error"


def test_predict_rejects_missing_field():
    with TestClient(create_app(MODEL_DIR)) as c:
        bad = {k: v for k, v in VALID.items() if k != "torque_nm"}
        r = c.post("/predict", json=bad)
        assert r.status_code == 422
        assert any(d["field"] == "torque_nm" for d in r.json()["error"]["details"])


def test_predict_503_without_model():
    with tempfile.TemporaryDirectory() as empty:
        with TestClient(create_app(empty)) as c:
            r = c.post("/predict", json=VALID)
            assert r.status_code == 503 and r.json()["error"]["code"] == "http_error"


def test_risk_threshold_env_override():
    os.environ["RISK_THRESHOLD_MEDIUM"] = "0.0001"
    os.environ["RISK_THRESHOLD_HIGH"] = "0.0002"
    try:
        with TestClient(create_app(MODEL_DIR)) as c:
            assert c.post("/predict", json=VALID).json()["risk_category"] in {"medium", "high"}
    finally:
        del os.environ["RISK_THRESHOLD_MEDIUM"], os.environ["RISK_THRESHOLD_HIGH"]


def _records(df):
    from app.schemas import COLUMN_MAP

    inv = {v: k for k, v in COLUMN_MAP.items()}
    return df[list(inv)].rename(columns=inv).to_dict("records")


def test_monitor_normal_batch_no_alert():
    batch = _records(make_synthetic_df(300, seed=11))
    with TestClient(create_app(MODEL_DIR)) as c:
        r = c.post("/monitor", json={"readings": batch})
        assert r.status_code == 200
        body = r.json()
        assert body["alert_level"] != "alert" and body["model_version"]
        assert body["monitoring_timestamp"] and "concept drift" in body["note"]


def test_monitor_shifted_batch_alerts():
    df = make_synthetic_df(300, seed=11)
    df["Torque [Nm]"] += 25
    df["Tool wear [min]"] += 80
    df["Air temperature [K]"] += 6
    with TestClient(create_app(MODEL_DIR)) as c:
        body = c.post("/monitor", json={"readings": _records(df)}).json()
        assert body["alert_level"] == "alert"
        assert "Torque [Nm]" in body["drifted_features"]


def test_monitor_rejects_empty_and_tiny_batches():
    with TestClient(create_app(MODEL_DIR)) as c:
        assert c.post("/monitor", json={"readings": []}).status_code == 422
        r = c.post("/monitor", json={"readings": [VALID]})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_metrics_are_safe_and_count_requests():
    with TestClient(create_app(MODEL_DIR)) as c:
        c.post("/predict", json=VALID)
        m = c.get("/metrics").json()
        assert m["requests_total"] >= 1 and sum(m["predictions_by_risk"].values()) == 1
        assert "300.1" not in str(m)  # no payload data


def test_docs_available():
    with TestClient(create_app(MODEL_DIR)) as c:
        assert c.get("/docs").status_code == 200
        assert c.get("/openapi.json").status_code == 200

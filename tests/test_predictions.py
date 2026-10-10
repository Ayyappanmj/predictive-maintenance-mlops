"""Model loading and inference tests."""

import tempfile
from pathlib import Path

from src.predict import load_model, predict_dataframe, risk_category
from src.train import train_pipeline
from tests.helpers import fast_config, make_synthetic_df

CFG = fast_config()


def _trained(tmp):
    train_pipeline(make_synthetic_df(), CFG, Path(tmp) / "m", Path(tmp) / "r", False)
    return load_model(Path(tmp) / "m")


def test_load_and_predict_schema():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _trained(tmp)
        out = predict_dataframe(bundle, make_synthetic_df(20, seed=5), CFG)
        assert list(out.columns) == [
            "failure_probability",
            "risk_category",
            "predicted_failure",
            "model_version",
        ]
        assert out["failure_probability"].between(0, 1).all()
        assert set(out["risk_category"]) <= {"low", "medium", "high"}


def test_missing_column_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _trained(tmp)
        bad = make_synthetic_df(5).drop(columns=["Torque [Nm]"])
        try:
            predict_dataframe(bundle, bad, CFG)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "Missing input columns" in str(e)


def test_invalid_value_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _trained(tmp)
        bad = make_synthetic_df(5)
        bad.loc[0, "Torque [Nm]"] = float("nan")
        try:
            predict_dataframe(bundle, bad, CFG)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "missing" in str(e)


def test_risk_category_boundaries():
    assert risk_category(0.1, 0.3, 0.6) == "low"
    assert risk_category(0.3, 0.3, 0.6) == "medium"
    assert risk_category(0.9, 0.3, 0.6) == "high"


def test_missing_model_files_raise():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_model(tmp)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass

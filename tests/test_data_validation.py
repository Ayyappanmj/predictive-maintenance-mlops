"""Data validation tests. Uses small synthetic frames shaped like AI4I 2020 (test data only)."""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import load_config
from src.data_validation import (
    assert_no_leakage,
    drop_leaky_columns,
    select_features_target,
    validate_dataframe,
    validate_sensor_values,
)

CFG = load_config()


def make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": rng.normal(40, 10, n).clip(3, 80),
            "Tool wear [min]": rng.integers(0, 250, n),
            "Machine failure": (np.arange(n) % 20 == 0).astype(int),
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )
    return df


def test_valid_data_passes():
    rep = validate_dataframe(make_df(), CFG)
    assert rep.ok, rep.errors


def test_missing_column_fails():
    rep = validate_dataframe(make_df().drop(columns=["Torque [Nm]"]), CFG)
    assert not rep.ok and "Missing required columns" in rep.errors[0]


def test_missing_sensor_values_detected():
    df = make_df()
    df.loc[3, "Torque [Nm]"] = np.nan
    assert any("missing" in e for e in validate_sensor_values(df, CFG))


def test_invalid_sensor_values_detected():
    df = make_df()
    df.loc[0, "Air temperature [K]"] = -50.0
    assert any("outside plausible range" in e for e in validate_sensor_values(df, CFG))


def test_non_numeric_sensor_values_detected():
    df = make_df()
    df["Tool wear [min]"] = df["Tool wear [min]"].astype(str)
    assert any("non-numeric" in e for e in validate_sensor_values(df, CFG))


def test_bad_category_fails():
    df = make_df()
    df.loc[0, "Type"] = "X"
    assert not validate_dataframe(df, CFG).ok


def test_non_binary_target_fails():
    df = make_df()
    df.loc[0, "Machine failure"] = 2
    assert not validate_dataframe(df, CFG).ok


def test_single_class_target_fails():
    df = make_df()
    df["Machine failure"] = 0
    assert not validate_dataframe(df, CFG).ok


def test_too_few_rows_fails():
    assert not validate_dataframe(make_df(n=20), CFG).ok


def test_leaky_columns_removed_and_features_clean():
    df = make_df()
    cleaned = drop_leaky_columns(df, CFG)
    for col in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert col not in cleaned.columns
    X, y = select_features_target(df, CFG)
    assert "Machine failure" not in X.columns and len(X) == len(y)


def test_leakage_guard_raises():
    X, _ = select_features_target(make_df(), CFG)
    X["TWF"] = 0
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(X, CFG)

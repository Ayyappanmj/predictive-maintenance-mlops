"""Unit tests for metrics, threshold selection and engineered features (hand-checkable data)."""

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, select_threshold
from src.preprocessing import ENGINEERED, add_engineered_features


def test_compute_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert 0 < m["roc_auc"] <= 1 and 0 < m["pr_auc"] <= 1


def test_select_threshold_maximises_f1_on_given_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    thr = select_threshold(y, p)
    assert 0.3 < thr <= 0.6  # perfectly separates the classes


def test_select_threshold_respects_recall_floor():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.55, 0.5, 0.6, 0.9, 0.95])
    thr = select_threshold(y, p, min_recall=1.0)
    assert compute_metrics(y, p, thr)["recall"] == 1.0


def test_select_threshold_rejects_unknown_strategy():
    try:
        select_threshold(np.array([0, 1]), np.array([0.2, 0.8]), strategy="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_engineered_features_math():
    X = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [60.0],
            "Torque [Nm]": [10.0],
        }
    )
    out = add_engineered_features(X)
    assert out[ENGINEERED[0]].iloc[0] == 10.0
    assert abs(out[ENGINEERED[1]].iloc[0] - 10.0 * 2 * np.pi) < 1e-9  # 60 rpm = 1 rev/s
    assert ENGINEERED[0] not in X.columns  # input not mutated

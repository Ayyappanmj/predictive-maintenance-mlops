"""Feature engineering and preprocessing. Transformers are fitted on training data only
(they live inside the sklearn Pipeline, which is fitted via pipeline.fit(X_train, y_train))."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

AIR, PROC = "Air temperature [K]", "Process temperature [K]"
SPEED, TORQUE = "Rotational speed [rpm]", "Torque [Nm]"
ENGINEERED = ["Temperature difference [K]", "Mechanical power [W]"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise physical features (no fitting, so no leakage)."""
    X = X.copy()
    X[ENGINEERED[0]] = X[PROC] - X[AIR]
    X[ENGINEERED[1]] = X[TORQUE] * X[SPEED] * 2 * np.pi / 60
    return X


def build_preprocessor(cfg: dict) -> Pipeline:
    d = cfg["data"]
    numeric = [*d["numeric_features"], *ENGINEERED]
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), d["categorical_features"]),
        ]
    )
    return Pipeline([("engineer", FunctionTransformer(add_engineered_features)), ("columns", ct)])


def input_columns(cfg: dict) -> list[str]:
    d = cfg["data"]
    return [*d["numeric_features"], *d["categorical_features"]]

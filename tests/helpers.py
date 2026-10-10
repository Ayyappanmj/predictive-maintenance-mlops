"""Synthetic test data with a learnable signal (for tests only; not the real dataset)."""

import copy

import numpy as np
import pandas as pd

from src.data_ingestion import load_config


def make_synthetic_df(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    torque = rng.normal(40, 10, n).clip(3, 80)
    wear = rng.integers(0, 250, n)
    risk = (torque - 40) / 10 + (wear - 125) / 60
    fail = (risk + rng.normal(0, 0.7, n) > 2.2).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": fail,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )


def fast_config() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["training"]["random_forest"]["n_estimators"] = 40
    return cfg

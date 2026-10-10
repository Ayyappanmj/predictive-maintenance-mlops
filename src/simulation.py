"""Reproducible SIMULATED sensor stream for demonstrating drift monitoring.

The AI4I 2020 dataset has no real timestamps. Every timestamp produced here is SIMULATED and
must never be presented as an original measurement. Batches are bootstrap resamples of the
reference data; from `drift_start` onward a progressive, injected shift is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import input_columns

SIM_START = pd.Timestamp("2026-01-01T00:00:00")
# injected shift per sensor, in units of the reference standard deviation (times strength)
SHIFT_PROFILE = {"Torque [Nm]": 1.0, "Tool wear [min]": 1.0, "Air temperature [K]": 0.5}


def shift_batch(
    batch: pd.DataFrame, reference: pd.DataFrame, cfg: dict, strength: float, ramp: float = 1.0
) -> pd.DataFrame:
    out = batch.copy()
    for col, mult in SHIFT_PROFILE.items():
        lo, hi = cfg["validation"]["ranges"][col]
        out[col] = (out[col] + mult * strength * ramp * reference[col].std()).clip(lo, hi)
    return out


def simulate_stream(
    reference: pd.DataFrame,
    cfg: dict,
    n_batches: int = 12,
    batch_size: int = 200,
    drift_start: int = 6,
    drift_strength: float = 2.0,
    seed: int = 42,
    batch_interval_minutes: int = 60,
    ramp_batches: int = 3,
) -> pd.DataFrame:
    """Return a long frame: batch_id, simulated_timestamp, injected_shift + input columns."""
    if not 0 <= drift_start <= n_batches:
        raise ValueError("drift_start must be between 0 and n_batches")
    rng = np.random.default_rng(seed)
    cols = input_columns(cfg)
    ref = reference[cols].reset_index(drop=True)
    parts = []
    for b in range(n_batches):
        idx = rng.integers(0, len(ref), batch_size)
        batch = ref.iloc[idx].reset_index(drop=True)
        shifted = b >= drift_start
        if shifted:
            batch = shift_batch(
                batch, ref, cfg, drift_strength, min(1.0, (b - drift_start + 1) / ramp_batches)
            )
        t0 = SIM_START + pd.Timedelta(minutes=b * batch_interval_minutes)
        step = pd.Timedelta(minutes=batch_interval_minutes) / batch_size
        batch.insert(0, "simulated_timestamp", [t0 + i * step for i in range(batch_size)])
        batch.insert(0, "batch_id", b)
        batch["injected_shift"] = shifted
        parts.append(batch)
    return pd.concat(parts, ignore_index=True)

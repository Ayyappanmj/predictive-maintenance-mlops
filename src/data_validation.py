"""Data validation: schema, types, ranges, target and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def required_columns(cfg: dict) -> list[str]:
    d = cfg["data"]
    return [*d["numeric_features"], *d["categorical_features"], d["target"]]


def validate_sensor_values(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Check numeric sensor columns for non-numeric, missing and out-of-range values.

    Reusable for training data and (later) incoming API/monitoring batches.
    """
    errors: list[str] = []
    ranges = cfg["validation"]["ranges"]
    for col in cfg["data"]["numeric_features"]:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col}: non-numeric values present")
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            errors.append(f"{col}: {n_missing} missing values")
        lo, hi = ranges[col]
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            errors.append(f"{col}: {bad} values outside plausible range [{lo}, {hi}]")
    return errors


def validate_dataframe(df: pd.DataFrame, cfg: dict) -> ValidationReport:
    rep = ValidationReport()
    d, v = cfg["data"], cfg["validation"]

    missing = [c for c in required_columns(cfg) if c not in df.columns]
    if missing:
        rep.errors.append(f"Missing required columns: {missing}")
        return rep  # nothing else is meaningful

    rep.stats["n_rows"] = int(len(df))
    if len(df) < v["min_rows"]:
        rep.errors.append(f"Only {len(df)} rows; need at least {v['min_rows']}")

    rep.errors += validate_sensor_values(df, cfg)

    for col in d["categorical_features"]:
        if df[col].isna().any():
            rep.errors.append(f"{col}: missing values")
        bad = set(df[col].dropna().unique()) - set(v["allowed_types"])
        if bad:
            rep.errors.append(f"{col}: unexpected categories {sorted(map(str, bad))}")

    target = df[d["target"]]
    if target.isna().any():
        rep.errors.append("Target contains missing values")
    elif not set(target.unique()) <= {0, 1}:
        rep.errors.append(f"Target must be binary 0/1, found {sorted(target.unique())[:5]}")
    else:
        rate = float(target.mean())
        rep.stats["failure_rate"] = rate
        if target.nunique() < 2:
            rep.errors.append("Target has only one class")
        elif rate < 0.005 or rate > 0.5:
            rep.warnings.append(f"Unusual failure rate {rate:.3%}")

    n_dup = int(df.duplicated().sum())
    rep.stats["n_duplicate_rows"] = n_dup
    if n_dup:
        rep.warnings.append(f"{n_dup} duplicate rows")
    return rep


def drop_leaky_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop identifiers and failure-mode flags (TWF/HDF/PWF/OSF/RNF), which define the target."""
    return df.drop(columns=[c for c in cfg["data"]["drop_columns"] if c in df.columns])


def assert_no_leakage(features: pd.DataFrame, cfg: dict) -> None:
    d = cfg["data"]
    forbidden = {d["target"], *d["drop_columns"]}
    leaked = sorted(forbidden & set(features.columns))
    if leaked:
        raise ValueError(f"Target leakage: forbidden columns in features: {leaked}")


def select_features_target(df: pd.DataFrame, cfg: dict):
    """Return (X, y) using only configured features; guarantees no leakage."""
    d = cfg["data"]
    X = df[[*d["numeric_features"], *d["categorical_features"]]].copy()
    assert_no_leakage(X, cfg)
    return X, df[d["target"]].astype(int)

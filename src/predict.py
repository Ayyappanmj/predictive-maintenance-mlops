"""Model loading and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.data_ingestion import ROOT, load_config
from src.data_validation import validate_sensor_values


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict


def load_model(model_dir=None) -> ModelBundle:
    model_dir = Path(model_dir or ROOT / "models")
    model_path, meta_path = model_dir / "model.joblib", model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. Run `python -m src.train`."
        )
    return ModelBundle(joblib.load(model_path), json.loads(meta_path.read_text()))


def risk_category(p: float, medium: float, high: float) -> str:
    return "high" if p >= high else "medium" if p >= medium else "low"


def predict_dataframe(
    bundle: ModelBundle, df: pd.DataFrame, cfg: dict | None = None, risk: dict | None = None
) -> pd.DataFrame:
    """Return failure probability, risk category and thresholded flag per row."""
    cfg = cfg or load_config()
    cols = bundle.metadata["input_schema"]["columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    X = df[cols].copy()
    errors = validate_sensor_values(X, cfg)
    if errors:
        raise ValueError("; ".join(errors))
    proba = bundle.pipeline.predict_proba(X)[:, 1]
    risk = risk or bundle.metadata["risk_thresholds"]
    return pd.DataFrame(
        {
            "failure_probability": proba,
            "risk_category": [risk_category(p, risk["medium"], risk["high"]) for p in proba],
            "predicted_failure": proba >= bundle.metadata["decision_threshold"],
            "model_version": bundle.metadata["model_version"],
        }
    )

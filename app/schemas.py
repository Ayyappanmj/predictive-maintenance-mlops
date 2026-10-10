"""Pydantic request/response models. Validation bounds come from configs/model_config.yaml."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.api_contract import COLUMN_MAP, EXAMPLE_READING
from src.data_ingestion import load_config

_R = load_config()["validation"]["ranges"]
DISCLAIMER = "Model estimate only. This is not a confirmed equipment failure."


def _f(col: str, desc: str):
    lo, hi = _R[col]
    return Field(..., ge=lo, le=hi, allow_inf_nan=False, description=desc)


class SensorReading(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_READING},
    )
    air_temperature_k: float = _f("Air temperature [K]", "Air temperature in Kelvin")
    process_temperature_k: float = _f("Process temperature [K]", "Process temperature in Kelvin")
    rotational_speed_rpm: float = _f("Rotational speed [rpm]", "Rotational speed in rpm")
    torque_nm: float = _f("Torque [Nm]", "Torque in Newton-metres")
    tool_wear_min: float = _f("Tool wear [min]", "Tool wear in minutes")
    product_type: Literal["L", "M", "H"] = Field(..., description="Product quality variant")


def readings_to_frame(readings: list[SensorReading]) -> pd.DataFrame:
    rows = [r.model_dump() for r in readings]
    return pd.DataFrame(rows).rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    failure_probability: float = Field(..., ge=0, le=1)
    risk_category: Literal["low", "medium", "high"]
    predicted_failure: bool = Field(..., description="Probability >= the model decision threshold")
    decision_threshold: float
    model_version: str
    prediction_timestamp: datetime
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_name: str
    trained_at_utc: str
    decision_threshold: float
    risk_thresholds: dict
    input_schema: dict
    validation_metrics: dict
    test_metrics: dict
    data_sha256: str
    model_sha256: str
    dataset_note: str
    disclaimer: str


class MonitorRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=5000)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class MonitorResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    monitoring_timestamp: str
    model_version: str | None = None
    n_reference: int
    n_current: int
    n_features: int
    n_drifted: int
    drift_share: float
    dataset_drift: bool
    alert_level: Literal["none", "warning", "alert"]
    drifted_features: list[str]
    features: list[dict]
    thresholds: dict
    note: str

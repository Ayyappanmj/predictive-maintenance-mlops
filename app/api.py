"""FastAPI backend. Run from the repo root: uvicorn app.api:app --reload"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import (
    DISCLAIMER,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitorRequest,
    MonitorResponse,
    PredictionResponse,
    SensorReading,
    readings_to_frame,
)
from src.data_ingestion import ROOT, load_config
from src.drift_monitor import detect_drift, load_reference
from src.predict import ModelBundle, load_model, predict_dataframe

logger = logging.getLogger("pm_api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging() -> None:
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.propagate = False


def log(msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


class Metrics:
    """In-memory operational counters. Never stores request payloads."""

    def __init__(self) -> None:
        self.start = time.time()
        self.lock = threading.Lock()
        self.requests: Counter = Counter()
        self.predictions: Counter = Counter()
        self.monitor: Counter = Counter()
        self.total_latency_ms = 0.0
        self.n = 0

    def record_request(self, path: str, status: int, ms: float) -> None:
        with self.lock:
            self.requests[f"{path} {status}"] += 1
            self.total_latency_ms += ms
            self.n += 1

    def record_prediction(self, risk: str) -> None:
        with self.lock:
            self.predictions[risk] += 1

    def record_monitor(self, level: str) -> None:
        with self.lock:
            self.monitor[level] += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "uptime_seconds": round(time.time() - self.start, 1),
                "requests_total": self.n,
                "requests_by_endpoint_status": dict(self.requests),
                "predictions_by_risk": dict(self.predictions),
                "drift_checks_by_alert_level": dict(self.monitor),
                "avg_latency_ms": round(self.total_latency_ms / self.n, 2) if self.n else 0.0,
            }


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message, "details": details})
    return JSONResponse(status_code=status, content=body.model_dump())


def _risk_thresholds(bundle: ModelBundle) -> dict:
    """Metadata defaults, optionally overridden by RISK_THRESHOLD_* environment variables."""
    risk = dict(bundle.metadata["risk_thresholds"])
    med = float(os.getenv("RISK_THRESHOLD_MEDIUM", risk["medium"]))
    high = float(os.getenv("RISK_THRESHOLD_HIGH", risk["high"]))
    if not 0 < med < high < 1:
        raise ValueError("Risk thresholds must satisfy 0 < medium < high < 1")
    return {"medium": med, "high": high}


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    configure_logging()
    cfg = load_config()
    metrics = Metrics()

    def resolve_dir() -> Path:
        d = model_dir or os.getenv("MODEL_DIR") or "models"
        d = Path(d)
        return d if d.is_absolute() else ROOT / d

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.bundle = load_model(resolve_dir())
            log("model_loaded", version=app.state.bundle.metadata["model_version"])
            try:
                app.state.reference = load_reference(resolve_dir())
            except FileNotFoundError:
                app.state.reference = None
                logger.warning("reference_data_not_found", extra={"extra_fields": {}})
        except FileNotFoundError as e:
            app.state.bundle = None
            app.state.reference = None
            logger.warning("model_not_found: %s", e, extra={"extra_fields": {}})
        yield

    app = FastAPI(
        title="Predictive Maintenance API",
        version="1.0.0",
        description=(
            "Estimates machine failure probability from sensor readings. "
            "Trained on the synthetic AI4I 2020 dataset. " + DISCLAIMER
        ),
        lifespan=lifespan,
    )
    app.state.bundle = None
    app.state.reference = None

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        metrics.record_request(path, response.status_code, ms)
        log(
            "request",
            request_id=rid,
            method=request.method,
            path=path,
            status=response.status_code,
            duration_ms=round(ms, 2),
        )
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]  # input values are not echoed
        return _error(422, "validation_error", "Invalid request payload.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.error("unhandled_error", exc_info=exc, extra={"extra_fields": {}})
        return _error(500, "internal_error", "Internal server error.")

    def require_model() -> ModelBundle:
        bundle = app.state.bundle
        if bundle is None:
            raise StarletteHTTPException(503, "Model not loaded. Train or deploy a model first.")
        return bundle

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        b = app.state.bundle
        return HealthResponse(
            status="ok" if b else "degraded",
            model_loaded=b is not None,
            model_version=b.metadata["model_version"] if b else None,
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info():
        m = require_model().metadata
        return ModelInfoResponse(
            model_version=m["model_version"],
            model_name=m["model_name"],
            trained_at_utc=m["trained_at_utc"],
            decision_threshold=m["decision_threshold"],
            risk_thresholds=_risk_thresholds(app.state.bundle),
            input_schema=m["input_schema"],
            validation_metrics=m["validation_metrics"],
            test_metrics=m["test_metrics"],
            data_sha256=m["data_sha256"],
            model_sha256=m["model_sha256"],
            dataset_note=m["dataset_note"],
            disclaimer=DISCLAIMER,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["model"],
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict(reading: SensorReading):
        bundle = require_model()
        try:
            out = predict_dataframe(
                bundle, readings_to_frame([reading]), cfg, risk=_risk_thresholds(bundle)
            ).iloc[0]
        except ValueError as e:
            return _error(422, "invalid_input", str(e))
        metrics.record_prediction(out["risk_category"])
        return PredictionResponse(
            failure_probability=float(out["failure_probability"]),
            risk_category=out["risk_category"],
            predicted_failure=bool(out["predicted_failure"]),
            decision_threshold=bundle.metadata["decision_threshold"],
            model_version=bundle.metadata["model_version"],
            prediction_timestamp=datetime.now(timezone.utc),
        )

    @app.post(
        "/monitor",
        response_model=MonitorResponse,
        tags=["monitoring"],
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def monitor(batch: MonitorRequest):
        """Data-drift check of a sensor batch against the training reference data.
        Data drift is not concept drift: it does not prove accuracy declined."""
        bundle = require_model()
        if app.state.reference is None:
            raise StarletteHTTPException(503, "Reference data not available next to the model.")
        try:
            result = detect_drift(
                app.state.reference,
                readings_to_frame(batch.readings),
                cfg,
                bundle.metadata["model_version"],
            )
        except ValueError as e:
            return _error(422, "invalid_input", str(e))
        metrics.record_monitor(result["alert_level"])
        log(
            "drift_check",
            alert_level=result["alert_level"],
            n_drifted=result["n_drifted"],
            n_current=result["n_current"],
        )
        return result

    @app.get("/metrics", tags=["system"])
    def get_metrics():
        snap = metrics.snapshot()
        b = app.state.bundle
        snap.update(
            model_loaded=b is not None, model_version=b.metadata["model_version"] if b else None
        )
        return snap

    return app


app = create_app()

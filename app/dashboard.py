"""Streamlit dashboard. Run from the repo root: streamlit run app/dashboard.py

Predictions come from the real trained model (local) or the real FastAPI service (API mode).
Anything simulated is labelled as such.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from app.schemas import COLUMN_MAP  # noqa: E402
from src.data_ingestion import load_config, load_raw  # noqa: E402
from src.drift_monitor import detect_drift, load_reference, render_html_report  # noqa: E402
from src.predict import ModelBundle, load_model, predict_dataframe  # noqa: E402
from src.preprocessing import input_columns  # noqa: E402
from src.simulation import simulate_stream  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

CFG = load_config()
NUM = CFG["data"]["numeric_features"]
TARGET = CFG["data"]["target"]
RANGES = CFG["validation"]["ranges"]
MODEL_DIR = os.getenv("MODEL_DIR", "models")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DISCLAIMER = "Model estimate only. This is not a confirmed equipment failure."

# Example starting values for the form (not dataset statistics).
DEFAULTS = {
    "in_air": 300.0,
    "in_proc": 310.0,
    "in_speed": 1500.0,
    "in_torque": 40.0,
    "in_wear": 100.0,
    "in_type": "M",
}
FORM_KEYS = {  # session key -> training column
    "in_air": "Air temperature [K]",
    "in_proc": "Process temperature [K]",
    "in_speed": "Rotational speed [rpm]",
    "in_torque": "Torque [Nm]",
    "in_wear": "Tool wear [min]",
    "in_type": "Type",
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

st.markdown(
    """<style>
    .block-container {padding-top: 1.5rem;}
    .risk-box {padding: 1rem 1.25rem; border-radius: 10px; font-size: 1.3rem; font-weight: 600;}
    </style>""",
    unsafe_allow_html=True,
)


# ---------- cached loaders ----------
@st.cache_resource(show_spinner=False)
def get_bundle(model_dir: str) -> ModelBundle | None:
    d = Path(model_dir)
    d = d if d.is_absolute() else ROOT / d
    try:
        return load_model(d)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_reference() -> pd.DataFrame | None:
    try:
        return load_raw(CFG)
    except FileNotFoundError:
        return None


# ---------- prediction helpers ----------
def input_frame() -> pd.DataFrame:
    return pd.DataFrame([{col: st.session_state[k] for k, col in FORM_KEYS.items()}])


def predict_via_api(df: pd.DataFrame) -> dict:
    inv = {v: k for k, v in COLUMN_MAP.items()}
    row = df.iloc[0]
    payload = {inv[c]: (str(row[c]) if c == "Type" else float(row[c])) for c in df.columns}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
    if r.status_code != 200:
        msg = r.json().get("error", {}).get("message", r.text) if r.content else r.reason
        raise RuntimeError(f"API returned {r.status_code}: {msg}")
    b = r.json()
    return {
        "failure_probability": b["failure_probability"],
        "risk_category": b["risk_category"],
        "predicted_failure": b["predicted_failure"],
        "model_version": b["model_version"],
        "timestamp": b["prediction_timestamp"],
        "source": f"API ({API_URL})",
    }


def predict_locally(bundle: ModelBundle, df: pd.DataFrame) -> dict:
    out = predict_dataframe(bundle, df, CFG).iloc[0]
    return {
        "failure_probability": float(out["failure_probability"]),
        "risk_category": out["risk_category"],
        "predicted_failure": bool(out["predicted_failure"]),
        "model_version": out["model_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "local model",
    }


def reset_form() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)


def gauge(prob: float, med: float, high: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Estimated failure probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f5a623"},
                "steps": [
                    {"range": [0, med * 100], "color": "#1f6f43"},
                    {"range": [med * 100, high * 100], "color": "#8a6d1d"},
                    {"range": [high * 100, 100], "color": "#8b2b2b"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# ---------- page ----------
st.title("🛠️ Predictive Maintenance Dashboard")
st.caption(
    "Industrial equipment failure risk from sensor data · "
    "AI4I 2020 synthetic dataset · demo project"
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Prediction source",
        ["Local model", "FastAPI service"],
        help="Local loads models/model.joblib. FastAPI calls API_URL/predict.",
    )
    if mode == "FastAPI service":
        st.caption(f"API: {API_URL}  (free hosting can take ~1 min to wake up)")
    if st.button("Reload model and data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

bundle = get_bundle(MODEL_DIR)
if bundle is None:
    st.error(
        f"No trained model found in `{MODEL_DIR}/`. "
        "Run `python scripts/download_data.py` then `python -m src.train`, "
        "and use 'Reload model and data' in the sidebar."
    )
    st.stop()
meta = bundle.metadata
risk = meta["risk_thresholds"]

with st.sidebar:
    st.divider()
    st.caption(f"Model: **{meta['model_name']}**")
    st.caption(f"Version: `{meta['model_version']}`")

tab_over, tab_pred, tab_data, tab_perf, tab_mon = st.tabs(
    ["Overview", "Predict", "Data explorer", "Model performance", "Monitoring"]
)

# ---- Overview ----
with tab_over:
    st.subheader("What this project does")
    st.write(
        "A machine learning model estimates the **probability that a machine fails** from "
        "five sensor readings and the product type. The pipeline is reproducible, the API is "
        "validated, and retraining and drift monitoring are automated (later project phases)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", meta["model_name"].replace("_", " ").title())
    c2.metric("Model version", meta["model_version"])
    c3.metric("Decision threshold", f"{meta['decision_threshold']:.3f}")
    st.info(
        "**Dataset:** AI4I 2020 Predictive Maintenance (UCI, CC BY 4.0). The data is "
        "**synthetic**, so results do not show performance on real equipment.  \n"
        "**Scope:** the model outputs a failure *probability*. It does not predict the time "
        "remaining before a failure.  \n" + DISCLAIMER
    )

# ---- Predict ----
with tab_pred:
    st.subheader("Sensor input")
    st.caption(
        "Starting values are examples, not recommendations. Valid ranges come from the "
        "project configuration."
    )
    with st.form("sensor_form"):
        c1, c2 = st.columns(2)
        for key, col, box in [
            ("in_air", NUM[0], c1),
            ("in_proc", NUM[1], c2),
            ("in_speed", NUM[2], c1),
            ("in_torque", NUM[3], c2),
            ("in_wear", NUM[4], c1),
        ]:
            lo, hi = RANGES[col]
            box.number_input(col, min_value=float(lo), max_value=float(hi), key=key, step=0.1)
        c2.selectbox("Product type", CFG["validation"]["allowed_types"], key="in_type")
        submitted = st.form_submit_button("Predict failure risk")
    st.button("Reset inputs", on_click=reset_form)

    if submitted:
        st.session_state.pop("last_error", None)
        try:
            frame = input_frame()
            with st.spinner("Predicting..."):
                if mode == "Local model":
                    st.session_state["last_result"] = predict_locally(bundle, frame)
                else:
                    st.session_state["last_result"] = predict_via_api(frame)
        except ValueError as e:
            st.session_state["last_error"] = f"Invalid input: {e}"
        except requests.RequestException as e:
            st.session_state["last_error"] = (
                f"Could not reach the API at {API_URL}. Is it running or waking up? ({e})"
            )
        except Exception as e:  # keep the dashboard alive on unexpected errors
            st.session_state["last_error"] = f"Prediction failed: {e}"

    if "last_error" in st.session_state:
        st.error(st.session_state["last_error"])
    res = st.session_state.get("last_result")
    if res:
        left, right = st.columns([1, 1])
        left.plotly_chart(gauge(res["failure_probability"], risk["medium"], risk["high"]))
        colour = {"low": "#1f6f43", "medium": "#8a6d1d", "high": "#8b2b2b"}[res["risk_category"]]
        right.markdown(
            f'<div class="risk-box" style="background:{colour}">'
            f"Risk category: {res['risk_category'].upper()}</div>",
            unsafe_allow_html=True,
        )
        right.write(
            f"Probability above decision threshold "
            f"({meta['decision_threshold']:.3f}): **{res['predicted_failure']}**"
        )
        right.caption(
            f"Model version `{res['model_version']}` · {res['timestamp']} · source: {res['source']}"
        )
        st.warning(DISCLAIMER)

# ---- Data explorer ----
with tab_data:
    ref = get_reference()
    if ref is None:
        st.info("Dataset not found. Run `python scripts/download_data.py` to enable this tab.")
    else:
        st.subheader("Sensor distributions")
        st.caption(
            "Source: the AI4I 2020 dataset (synthetic). Densities are normalised because "
            "failures are rare."
        )
        col = st.selectbox("Sensor", NUM)
        plot_df = ref.assign(Outcome=ref[TARGET].map({0: "No failure", 1: "Failure"}))
        fig = px.histogram(
            plot_df,
            x=col,
            color="Outcome",
            barmode="overlay",
            opacity=0.6,
            histnorm="probability density",
            nbins=40,
            color_discrete_map={"No failure": "#4c78a8", "Failure": "#e45756"},
        )
        if "last_result" in st.session_state:
            key = [k for k, c in FORM_KEYS.items() if c == col][0]
            fig.add_vline(
                x=st.session_state[key],
                line_dash="dash",
                line_color="#f5a623",
                annotation_text="current input",
            )
        st.plotly_chart(fig)
        c1, c2 = st.columns(2)
        counts = ref[TARGET].value_counts().sort_index()
        c1.plotly_chart(
            px.bar(
                x=["No failure", "Failure"],
                y=counts.values,
                labels={"x": "", "y": "Rows"},
                title="Class balance",
            )
        )
        by_type = ref.groupby("Type")[TARGET].mean().reset_index()
        c2.plotly_chart(
            px.bar(
                by_type,
                x="Type",
                y=TARGET,
                title="Failure rate by product type",
                labels={TARGET: "Failure rate"},
            )
        )

# ---- Model performance ----
with tab_perf:
    st.subheader("Evaluation results")
    st.caption(
        "Models are compared on validation data. The test set was used once, after the "
        "final model and threshold were chosen."
    )
    rows = []
    for name, m in meta["validation_metrics"].items():
        rows.append(
            {
                "model": name,
                "split": "validation",
                **{
                    k: m[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "threshold")
                },
            }
        )
    t = meta["test_metrics"]
    rows.append(
        {
            "model": meta["model_name"],
            "split": "TEST (final)",
            **{k: t[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "threshold")},
        }
    )
    st.dataframe(pd.DataFrame(rows).round(3))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test precision", f"{t['precision']:.3f}")
    c2.metric("Test recall", f"{t['recall']:.3f}")
    c3.metric("Test PR-AUC", f"{t['pr_auc']:.3f}")
    c4.metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
    st.caption(
        f"Test set: {t['n_samples']} rows, {t['n_positive']} failures. "
        "With few positives, metrics are noisy."
    )
    c1, c2 = st.columns(2)
    cm = t["confusion_matrix"]
    c1.plotly_chart(
        px.imshow(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            text_auto=True,
            x=["No failure", "Failure"],
            y=["No failure", "Failure"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion matrix (test)",
            color_continuous_scale="Blues",
        )
    )
    imp = pd.DataFrame(
        [
            {"feature": k, "importance": v["mean"], "std": v["std"]}
            for k, v in meta["permutation_importance"].items()
        ]
    )
    c2.plotly_chart(
        px.bar(
            imp.sort_values("importance"),
            x="importance",
            y="feature",
            error_x="std",
            orientation="h",
            title="Permutation importance (validation)",
        )
    )
    with st.expander("Training metadata"):
        st.json(
            {
                k: meta[k]
                for k in (
                    "model_version",
                    "trained_at_utc",
                    "model_name",
                    "seed",
                    "split_sizes",
                    "positives",
                    "versions",
                    "data_sha256",
                    "model_sha256",
                )
            }
        )
    st.download_button(
        "Download full model metadata (JSON)",
        json.dumps(meta, indent=2),
        file_name=f"model_metadata_{meta['model_version']}.json",
        mime="application/json",
    )


# ---- Monitoring ----
@st.cache_data(show_spinner=False)
def get_monitor_reference(model_dir: str) -> pd.DataFrame | None:
    d = Path(model_dir)
    try:
        return load_reference(d if d.is_absolute() else ROOT / d)
    except FileNotFoundError:
        return None


def show_drift_result(result: dict, ref: pd.DataFrame, cur: pd.DataFrame, key: str) -> None:
    msg = (
        f"Alert level: {result['alert_level'].upper()} · {result['n_drifted']}/"
        f"{result['n_features']} features drifted ({result['drift_share']:.0%}) · "
        f"model `{result['model_version']}` · {result['monitoring_timestamp']}"
    )
    {"none": st.success, "warning": st.warning, "alert": st.error}[result["alert_level"]](msg)
    st.dataframe(pd.DataFrame(result["features"]).round(4))
    col = st.selectbox("Compare a sensor", NUM, key=f"cmp_{key}")
    both = pd.concat([ref[[col]].assign(dataset="reference"), cur[[col]].assign(dataset="current")])
    st.plotly_chart(
        px.histogram(
            both,
            x=col,
            color="dataset",
            barmode="overlay",
            opacity=0.6,
            histnorm="probability density",
            nbins=40,
        )
    )
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download drift report (JSON)",
        json.dumps(result, indent=2),
        file_name="drift_report.json",
        mime="application/json",
        key=f"dj_{key}",
    )
    c2.download_button(
        "Download drift report (HTML)",
        render_html_report(result),
        file_name="drift_report.html",
        mime="text/html",
        key=f"dh_{key}",
    )
    st.caption(result["note"])


with tab_mon:
    st.subheader("Data drift monitoring")
    st.info(
        "**Data drift** = input distributions changed versus the training data. "
        "**Concept drift** = the input-to-failure relationship changed. Data drift alone does "
        "not prove accuracy dropped, and measuring accuracy needs reliable failure labels."
    )
    ref_mon = get_monitor_reference(MODEL_DIR)
    if ref_mon is None:
        st.warning(
            "Reference data (`reference.csv`) not found next to the model. "
            "Re-run `python -m src.train`, then reload from the sidebar."
        )
    else:
        st.caption(
            f"Reference: {len(ref_mon)} training rows. Tests: KS (numeric), chi-square "
            f"(categorical), alpha={CFG['drift']['alpha']}, alert when "
            f"≥{CFG['drift']['alert_share']:.0%} of features drift."
        )
        st.markdown("#### 1. Check a current sensor batch")
        up = st.file_uploader("Current batch (CSV with the sensor columns and Type)", type="csv")
        if up is not None:
            try:
                cur = pd.read_csv(up)
                res = detect_drift(ref_mon, cur, CFG, meta["model_version"])
                show_drift_result(res, ref_mon, cur[input_columns(CFG)], "upload")
            except ValueError as e:
                st.error(f"Cannot analyse this batch: {e}")
            except Exception as e:
                st.error(f"Could not read or analyse the file: {e}")

        st.markdown("#### 2. Simulated sensor stream (SIMULATION)")
        st.warning(
            "Everything in this section is **simulated**: batches are resampled from the "
            "reference data and a shift is injected on purpose. Timestamps are simulated, "
            "not original measurements."
        )
        c1, c2, c3 = st.columns(3)
        n_b = c1.slider("Batches", 4, 30, 12)
        b_size = c2.slider("Rows per batch", 50, 500, 200, step=50)
        strength = c3.slider("Injected shift strength", 0.5, 4.0, 2.0, step=0.5)
        d_start = st.slider("Shift starts at batch", 1, n_b - 1, min(6, n_b - 1))
        seed = st.number_input("Random seed", 0, 10_000, 42)
        if st.button("Run simulation"):
            stream = simulate_stream(ref_mon, CFG, n_b, b_size, d_start, strength, int(seed))
            rows = []
            for b, g in stream.groupby("batch_id"):
                r = detect_drift(ref_mon, g[input_columns(CFG)], CFG, meta["model_version"])
                rows.append(
                    {
                        "batch": int(b),
                        "simulated_timestamp": g["simulated_timestamp"].min(),
                        "drift_share": r["drift_share"],
                        "alert_level": r["alert_level"],
                        "injected_shift": bool(g["injected_shift"].iloc[0]),
                    }
                )
            st.session_state["stream_data"] = stream
            st.session_state["stream_summary"] = pd.DataFrame(rows)
        if "stream_summary" in st.session_state:
            summ, stream = st.session_state["stream_summary"], st.session_state["stream_data"]
            st.plotly_chart(
                px.line(
                    summ,
                    x="batch",
                    y="drift_share",
                    markers=True,
                    color_discrete_sequence=["#f5a623"],
                    title="Share of drifted features per simulated batch",
                )
            )
            st.dataframe(summ)
            sensor = st.selectbox("Sensor mean per batch (simulated)", NUM, key="stream_sensor")
            st.plotly_chart(
                px.line(
                    stream.groupby("batch_id")[sensor].mean().reset_index(),
                    x="batch_id",
                    y=sensor,
                    markers=True,
                )
            )
            st.download_button(
                "Download simulated stream (CSV)",
                stream.to_csv(index=False),
                file_name="simulated_stream.csv",
                mime="text/csv",
            )

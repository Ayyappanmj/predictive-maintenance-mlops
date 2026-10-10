"""Training pipeline. Usage: python -m src.train"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_ingestion import ROOT, load_config, load_raw
from src.data_validation import select_features_target, validate_dataframe
from src.evaluate import (
    compute_metrics,
    permutation_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    select_threshold,
)
from src.preprocessing import ENGINEERED, build_preprocessor, input_columns

logger = logging.getLogger(__name__)


def split_data(X, y, cfg: dict):
    """Stratified train / validation / test split (test is held out until the very end)."""
    seed, s = cfg["seed"], cfg["split"]
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=s["test_size"], stratify=y, random_state=seed
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=s["val_size"], stratify=y_tv, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def build_models(cfg: dict) -> dict[str, Pipeline]:
    """Class imbalance is handled with class weights (no resampling, no extra dependency)."""
    t, seed = cfg["training"], cfg["seed"]
    lr, rf = t["logistic_regression"], t["random_forest"]
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(cfg)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=lr["max_iter"],
                        C=lr["C"],
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(cfg)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=rf["n_estimators"],
                        min_samples_leaf=rf["min_samples_leaf"],
                        max_depth=rf["max_depth"],
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def train_pipeline(
    df: pd.DataFrame, cfg: dict, model_dir=None, report_dir=None, make_plots: bool = True
) -> dict:
    """Train, select on validation, evaluate on test once. Returns the metadata dict."""
    t = cfg["training"]
    model_dir = Path(model_dir or ROOT / t["model_dir"])
    report_dir = Path(report_dir or ROOT / t["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"])

    rep = validate_dataframe(df, cfg)
    if not rep.ok:
        raise ValueError(f"Data validation failed: {rep.errors}")
    X, y = select_features_target(df, cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, cfg)

    # 1) Fit every candidate on TRAIN, score on VALIDATION.
    comparison, fitted = {}, {}
    for name, pipe in build_models(cfg).items():
        pipe.fit(X_tr, y_tr)
        proba_va = pipe.predict_proba(X_va)[:, 1]
        thr = select_threshold(
            y_va, proba_va, t["threshold_strategy"], t["min_recall_for_threshold"]
        )
        comparison[name] = compute_metrics(y_va, proba_va, thr)
        fitted[name] = pipe
        logger.info(
            "%s val PR-AUC=%.4f recall=%.3f",
            name,
            comparison[name]["pr_auc"],
            comparison[name]["recall"],
        )

    # 2) Choose final model on validation results only.
    metric = t["selection_metric"]
    best = max(comparison, key=lambda n: comparison[n][metric])
    model, threshold = fitted[best], comparison[best]["threshold"]

    # 3) Evaluate the chosen configuration on the untouched TEST set (once).
    test_metrics = compute_metrics(y_te, model.predict_proba(X_te)[:, 1], threshold)
    importance = permutation_feature_importance(model, X_va, y_va, cfg["seed"])

    trained_at = datetime.now(timezone.utc)
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    X_tr.reset_index(drop=True).to_csv(model_dir / "reference.csv", index=False)
    metadata = {
        "model_version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at_utc": trained_at.isoformat(),
        "model_name": best,
        "selection": {"metric": f"val_{metric}", "chosen_on": "validation"},
        "decision_threshold": threshold,
        "risk_thresholds": cfg["risk"],
        "seed": cfg["seed"],
        "input_schema": {
            "columns": input_columns(cfg),
            "numeric": cfg["data"]["numeric_features"],
            "categorical": cfg["data"]["categorical_features"],
            "engineered": ENGINEERED,
        },
        "split_sizes": {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)},
        "positives": {
            "train": int(y_tr.sum()),
            "validation": int(y_va.sum()),
            "test": int(y_te.sum()),
        },
        "validation_metrics": comparison,
        "test_metrics": test_metrics,
        "permutation_importance": importance,
        "reference_file": "reference.csv",
        "data_sha256": _data_hash(df),
        "model_sha256": _sha256_file(model_path),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_note": "AI4I 2020 is synthetic data (UCI, CC BY 4.0).",
        "disclaimer": "Outputs are model estimates, not confirmed equipment failures.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (report_dir / "evaluation.json").write_text(
        json.dumps({"validation": comparison, "test": test_metrics, "selected": best}, indent=2)
    )
    if make_plots:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            report_dir / "confusion_matrix.png",
            f"Test set: {best}",
        )
        plot_feature_importance(importance, report_dir / "feature_importance.png")
    return metadata


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train a candidate model.")
    ap.add_argument("--model-dir", default=None, help="where to write model artifacts")
    ap.add_argument("--report-dir", default=None, help="where to write evaluation reports")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    meta = train_pipeline(load_raw(cfg), cfg, args.model_dir, args.report_dir)
    print(f"Selected model: {meta['model_name']} (version {meta['model_version']})")
    for n, m in meta["validation_metrics"].items():
        print(
            f"  val  {n:20s} PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        )
    m = meta["test_metrics"]
    print(
        f"  TEST {meta['model_name']:20s} PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

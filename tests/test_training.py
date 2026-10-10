"""Training tests on synthetic data (fast)."""

import json
import tempfile
from pathlib import Path

from sklearn.pipeline import Pipeline

from src.data_validation import select_features_target
from src.preprocessing import build_preprocessor
from src.train import split_data, train_pipeline
from tests.helpers import fast_config, make_synthetic_df

CFG = fast_config()


def test_training_completes_and_writes_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        meta = train_pipeline(
            make_synthetic_df(), CFG, Path(tmp) / "m", Path(tmp) / "r", make_plots=False
        )
        assert (Path(tmp) / "m" / "model.joblib").exists()
        saved = json.loads((Path(tmp) / "m" / "metadata.json").read_text())
        assert saved["model_name"] in {"logistic_regression", "random_forest"}
        assert saved["selection"]["chosen_on"] == "validation"
        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            assert 0.0 <= meta["test_metrics"][k] <= 1.0
        assert len(saved["model_sha256"]) == 64


def test_training_is_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        a = train_pipeline(make_synthetic_df(), CFG, Path(tmp) / "a", Path(tmp) / "ra", False)
        b = train_pipeline(make_synthetic_df(), CFG, Path(tmp) / "b", Path(tmp) / "rb", False)
        for name, m in a["validation_metrics"].items():
            assert m["pr_auc"] == b["validation_metrics"][name]["pr_auc"]


def test_split_is_disjoint_and_stratified():
    X, y = select_features_target(make_synthetic_df(), CFG)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, CFG)
    assert not (set(X_tr.index) & set(X_va.index)) and not (set(X_tr.index) & set(X_te.index))
    assert not (set(X_va.index) & set(X_te.index))
    assert abs(y_tr.mean() - y_te.mean()) < 0.02


def test_features_exclude_leaky_columns():
    X, _ = select_features_target(make_synthetic_df(), CFG)
    assert not ({"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "UDI"} & set(X.columns))


def test_preprocessor_fits_on_train_only():
    X, y = select_features_target(make_synthetic_df(), CFG)
    X_tr, X_va, *_ = split_data(X, y, CFG)
    prep = build_preprocessor(CFG)
    prep.fit(X_tr)
    scaler = prep.named_steps["columns"].named_transformers_["num"]
    assert abs(scaler.mean_[0] - 300) < 1  # learned from train rows
    assert scaler.n_samples_seen_ == len(X_tr)
    assert isinstance(prep, Pipeline)

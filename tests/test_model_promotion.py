"""Promotion, rejection, rollback and bundle tests.

Candidates are real trained models (synthetic data). Where a scenario needs specific
validation metrics, the test edits metadata.json to construct it (scenario setup only).
"""

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from src.model_bundle import extract_bundle, fetch_bundle, package_bundle
from src.promote_model import (
    FILES,
    list_versions,
    promote_candidate,
    read_meta,
    rollback,
)
from src.train import train_pipeline
from tests.helpers import fast_config, make_synthetic_df

CFG = fast_config()


def _model(root: Path, name: str, pr_auc=0.9, recall=0.9) -> Path:
    d = root / name
    train_pipeline(make_synthetic_df(), CFG, d, root / f"{name}_reports", False)
    _edit(d, version=name, pr_auc=pr_auc, recall=recall)
    return d


def _edit(d: Path, version=None, pr_auc=None, recall=None, fn=None) -> None:
    meta = json.loads((d / "metadata.json").read_text())
    m = meta["validation_metrics"][meta["model_name"]]
    if version:
        meta["model_version"] = version
    if pr_auc is not None:
        m["pr_auc"] = pr_auc
    if recall is not None:
        m["recall"] = recall
    if fn:
        fn(meta)
    (d / "metadata.json").write_text(json.dumps(meta))


def _hashes(d: Path) -> dict:
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in FILES}


def _promote(cand, appr, reg, tests=True, **kw):
    return promote_candidate(cand, appr, reg, CFG, tests_passed=tests, **kw)


def test_first_model_promoted_when_none_approved():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cand = _model(t, "v1")
        d = _promote(cand, t / "approved", t / "reg")
        assert d["promoted"] and read_meta(t / "approved")["model_version"] == "v1"


def test_tampered_artifact_rejected_and_approved_untouched():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _promote(_model(t, "v1"), t / "approved", t / "reg")
        before = _hashes(t / "approved")
        bad = _model(t, "v2", pr_auc=0.99)
        (bad / "model.joblib").write_bytes(b"corrupted")
        d = _promote(bad, t / "approved", t / "reg")
        assert not d["promoted"]
        assert any("artifact_integrity" in r for r in d["rejection_reasons"])
        assert _hashes(t / "approved") == before


def test_failed_tests_block_promotion():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        d = _promote(_model(t, "v1"), t / "approved", t / "reg", tests=False)
        assert not d["promoted"] and not (t / "approved").exists()
        assert any("automated_tests" in r for r in d["rejection_reasons"])


def test_low_recall_and_low_pr_auc_rejected():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        d = _promote(_model(t, "v1", recall=0.2), t / "approved", t / "reg")
        assert not d["promoted"] and any("min_val_recall" in r for r in d["rejection_reasons"])
        d = _promote(_model(t, "v2", pr_auc=0.1), t / "approved", t / "reg")
        assert not d["promoted"] and any("min_val_pr_auc" in r for r in d["rejection_reasons"])


def test_threshold_overrides_apply():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cand = _model(t, "v1", pr_auc=0.7, recall=0.7)
        assert not _promote(cand, t / "a", t / "r", min_pr_auc=0.95)["promoted"]
        assert _promote(cand, t / "a", t / "r", min_pr_auc=0.6, min_recall=0.6)["promoted"]


def test_candidate_not_better_than_approved_rejected():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _promote(_model(t, "v1", pr_auc=0.95), t / "approved", t / "reg")
        d = _promote(_model(t, "v2", pr_auc=0.80), t / "approved", t / "reg")
        assert not d["promoted"]
        assert any("better_than_approved" in r for r in d["rejection_reasons"])
        assert read_meta(t / "approved")["model_version"] == "v1"


def test_better_candidate_promoted_and_previous_archived():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _promote(_model(t, "v1", pr_auc=0.80), t / "approved", t / "reg")
        d = _promote(_model(t, "v2", pr_auc=0.90), t / "approved", t / "reg")
        assert d["promoted"] and d["approved_version_before"] == "v1"
        assert read_meta(t / "approved")["model_version"] == "v2"
        assert list_versions(t / "approved", t / "reg") == {"approved": "v2", "archived": ["v1"]}


def test_input_schema_mismatch_rejected():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cand = _model(t, "v1")
        _edit(cand, fn=lambda m: m["input_schema"]["columns"].reverse())
        d = _promote(cand, t / "approved", t / "reg")
        assert not d["promoted"] and any("input_schema" in r for r in d["rejection_reasons"])


def test_rollback_restores_previous_model():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        v1 = _model(t, "v1", pr_auc=0.80)
        v1_hashes = _hashes(v1)
        _promote(v1, t / "approved", t / "reg")
        _promote(_model(t, "v2", pr_auc=0.90), t / "approved", t / "reg")
        rollback("v1", t / "approved", t / "reg")
        assert read_meta(t / "approved")["model_version"] == "v1"
        assert _hashes(t / "approved") == v1_hashes
        assert "v2" in list_versions(t / "approved", t / "reg")["archived"]


def test_rollback_unknown_version_raises():
    with tempfile.TemporaryDirectory() as t:
        try:
            rollback("nope", Path(t) / "a", Path(t) / "r")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass


def test_rejection_logged_with_reasons():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _promote(_model(t, "v1", recall=0.1), t / "approved", t / "reg")
        lines = (t / "reg" / "promotion_log.jsonl").read_text().strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["promoted"] is False and entry["rejection_reasons"]


def test_dry_run_changes_nothing():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        d = _promote(_model(t, "v1"), t / "approved", t / "reg", dry_run=True)
        assert d["would_promote"] and not d["promoted"]
        assert not (t / "approved").exists() and not (t / "reg").exists()


def test_missing_candidate_rejected():
    with tempfile.TemporaryDirectory() as t:
        d = _promote(Path(t) / "none", Path(t) / "approved", Path(t) / "reg")
        assert not d["promoted"]


def test_bundle_roundtrip_and_fetch_via_file_url():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        src = _model(t, "v1")
        bundle = package_bundle(src, t / "dist" / "bundle.tar.gz")
        assert fetch_bundle(bundle.as_uri(), t / "deployed")
        assert _hashes(t / "deployed") == _hashes(src)


def test_extraction_rejects_unexpected_files():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        evil = t / "evil.tar.gz"
        (t / "x.txt").write_text("x")
        with tarfile.open(evil, "w:gz") as tf:
            tf.add(t / "x.txt", arcname="../escape.txt")
        try:
            extract_bundle(evil, t / "out")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        assert not (t / "escape.txt").exists()


def test_package_refuses_corrupt_model():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        src = _model(t, "v1")
        (src / "model.joblib").write_bytes(b"bad")
        try:
            package_bundle(src, t / "b.tar.gz")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

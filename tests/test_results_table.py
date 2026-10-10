"""The README results generator must only report numbers present in the metadata."""

import tempfile
from pathlib import Path

from scripts.generate_results_table import END, START, results_markdown, update_readme

M = {
    "precision": 0.5,
    "recall": 0.75,
    "f1": 0.6,
    "pr_auc": 0.62,
    "roc_auc": 0.9,
    "threshold": 0.4,
    "n_samples": 100,
    "n_positive": 8,
    "confusion_matrix": {"tn": 80, "fp": 12, "fn": 2, "tp": 6},
}
META = {
    "model_version": "v1",
    "trained_at_utc": "2026-01-02T00:00:00+00:00",
    "seed": 42,
    "model_name": "random_forest",
    "test_metrics": M,
    "validation_metrics": {"logistic_regression": M, "random_forest": M},
}


def test_markdown_contains_metadata_numbers_only():
    md = results_markdown(META)
    assert "0.750" in md and "0.620" in md and "80 / 12 / 2 / 6" in md
    assert "random_forest (test, final)" in md and "synthetic" in md


def test_update_readme_replaces_only_marked_block():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "README.md"
        p.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
        update_readme(p, "NEW")
        text = p.read_text()
        assert "NEW" in text and "old" not in text and text.startswith("intro")
        assert text.endswith("outro\n")


def test_update_readme_requires_markers():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "README.md"
        p.write_text("no markers")
        try:
            update_readme(p, "x")
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass

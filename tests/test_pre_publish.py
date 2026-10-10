"""Tests for the pre-publish safety check."""

import tempfile
from pathlib import Path

from scripts.pre_publish_check import REQUIRED, check


def _make(root: Path, extra: dict | None = None) -> list[str]:
    files = {name: "x" for name in REQUIRED}
    files["LICENSE"] = "MIT License, Jane Doe"
    files.update(extra or {})
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return list(files)


def test_clean_repo_passes():
    with tempfile.TemporaryDirectory() as t:
        errors, warnings = check(Path(t), _make(Path(t)))
        assert not errors and not warnings


def test_env_model_and_data_files_blocked():
    with tempfile.TemporaryDirectory() as t:
        files = _make(
            Path(t),
            {
                ".env": "A=1",
                "models/model.joblib": "b",
                "data/ai4i2020.csv": "c",
                "reports/x.json": "{}",
            },
        )
        errors, _ = check(Path(t), files)
        text = " ".join(errors)
        assert ".env" in text and "model.joblib" in text and "ai4i2020.csv" in text
        assert "reports/x.json" in text


def test_folder_readmes_allowed():
    with tempfile.TemporaryDirectory() as t:
        files = _make(
            Path(t), {"data/README.md": "r", "models/README.md": "r", "reports/README.md": "r"}
        )
        assert not check(Path(t), files)[0]


def test_secret_detected():
    with tempfile.TemporaryDirectory() as t:
        token = "ghp_" + "a" * 36
        errors, _ = check(Path(t), _make(Path(t), {"notes.txt": f"token {token}"}))
        assert any("GitHub token" in e for e in errors)


def test_missing_required_file_and_license_placeholder():
    with tempfile.TemporaryDirectory() as t:
        files = _make(Path(t))
        (Path(t) / "LICENSE").write_text("Copyright YOUR NAME")
        (Path(t) / "Makefile").unlink()
        errors, warnings = check(Path(t), [f for f in files if f != "Makefile"])
        assert any("Makefile" in e for e in errors)
        assert any("YOUR NAME" in w for w in warnings)

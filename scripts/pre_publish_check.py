"""Check what would be committed before publishing to GitHub.

Usage: python scripts/pre_publish_check.py
Looks at files git would add (respecting .gitignore); falls back to walking the folder.
Exit code 1 if blocking problems are found.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
REQUIRED = [
    "app/api.py",
    "app/dashboard.py",
    "app/schemas.py",
    "src/train.py",
    "src/predict.py",
    "src/drift_monitor.py",
    "src/promote_model.py",
    "configs/model_config.yaml",
    ".github/workflows/ci.yml",
    ".github/workflows/retrain.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
    "Makefile",
    ".gitignore",
    ".env.example",
]
FORBIDDEN = [  # (regex on posix path, reason)
    (r"(^|/)\.env$", "environment file with possible secrets"),
    (r"\.(joblib|pkl|pickle|pem|key)$", "model binary or key file"),
    (r"^data/(?!README\.md$).+", "dataset file (download it instead)"),
    (r"^models/(?!README\.md$).+", "trained model artifact"),
    (r"^reports/(?!README\.md$).+", "generated report"),
    (r"^(model_registry|dist)/", "registry or build output"),
    (r"(^|/)__pycache__/", "Python cache"),
    (r"(^|/)(\.venv|venv)/", "virtual environment"),
]
SECRET_PATTERNS = [
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (
        r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]",
        "hard-coded secret",
    ),  # noqa: E501
]


def tracked_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [p for p in out.splitlines() if (root / p).exists()]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache"}
        return [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(root).parts)
        ]


def check(root: Path, files: list[str]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    fileset = set(files)
    for req in REQUIRED:
        if req not in fileset:
            errors.append(f"missing required file: {req}")
    for f in files:
        for pattern, reason in FORBIDDEN:
            if re.search(pattern, f):
                errors.append(f"should not be committed: {f} ({reason})")
        path = root / f
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"file larger than 5 MB: {f}")
            continue
        if path.suffix in {".py", ".yml", ".yaml", ".toml", ".md", ".txt", ".cfg", ".json", ""}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern, what in SECRET_PATTERNS:
                if re.search(pattern, text) and f != "scripts/pre_publish_check.py":
                    errors.append(f"possible {what} in {f}")
    if "LICENSE" in fileset and "YOUR NAME" in (root / "LICENSE").read_text(errors="ignore"):
        warnings.append("LICENSE still contains the placeholder 'YOUR NAME'")
    return errors, warnings


def main() -> int:
    files = tracked_files(ROOT)
    errors, warnings = check(ROOT, files)
    print(f"{len(files)} files would be committed")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("OK to publish" if not errors else "NOT ready to publish")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

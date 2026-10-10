"""Package and fetch model bundles (tar.gz of model.joblib, metadata.json, reference.csv).

Used to move an approved model from CI (GitHub Release asset) to a deployment whose local
filesystem is temporary. Extraction accepts ONLY the three expected files and verifies the
model hash recorded in metadata.json.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from src.promote_model import FILES, _install, check_artifact_integrity

MAX_FILE_BYTES = 200 * 1024 * 1024


def package_bundle(model_dir, out_path) -> Path:
    model_dir, out_path = Path(model_dir), Path(out_path)
    ok, detail = check_artifact_integrity(model_dir)
    if not ok:
        raise ValueError(f"Refusing to package: {detail}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        for f in FILES:
            tf.add(model_dir / f, arcname=f)
    return out_path


def extract_bundle(archive, dest_dir) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if sorted(m.name for m in members) != sorted(FILES):
            raise ValueError(f"Unexpected bundle contents: {[m.name for m in members]}")
        for m in members:
            if not m.isfile() or m.size > MAX_FILE_BYTES:
                raise ValueError(f"Unsafe bundle member: {m.name}")
            (dest_dir / m.name).write_bytes(tf.extractfile(m).read())
    return dest_dir


def fetch_bundle(
    url: str, dest_dir, allow_missing: bool = False, expected_sha256: str | None = None
) -> bool:
    """Download, verify and install a bundle into dest_dir. Returns True if installed."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
                archive.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404 and allow_missing:
                return False
            raise
        if expected_sha256 and hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Bundle sha256 does not match the expected value")
        extracted = extract_bundle(archive, Path(tmp) / "x")
        ok, detail = check_artifact_integrity(extracted)
        if not ok:
            raise ValueError(f"Downloaded model failed integrity check: {detail}")
        _install(extracted, Path(dest_dir))
    return True

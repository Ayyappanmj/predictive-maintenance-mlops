"""Fetch the latest approved model bundle (a GitHub Release asset) into models/.

Public repositories only. Examples:
  python scripts/fetch_model.py --repo OWNER/REPO
  python scripts/fetch_model.py --repo OWNER/REPO --allow-missing     (CI first run)
  python scripts/fetch_model.py --url file:///tmp/model-bundle.tar.gz (testing)
"""

import argparse
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_bundle import fetch_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="OWNER/REPO on GitHub")
    ap.add_argument("--tag", default="model-latest")
    ap.add_argument("--url", help="override the download URL")
    ap.add_argument("--dest", default="models")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 if no bundle has been published yet (HTTP 404)",
    )
    ap.add_argument("--expected-sha256", default=None)
    a = ap.parse_args()
    url = a.url or (
        f"https://github.com/{a.repo}/releases/download/{a.tag}/model-bundle.tar.gz"
        if a.repo
        else None
    )
    if not url:
        ap.error("provide --repo or --url")
    try:
        installed = fetch_bundle(url, a.dest, a.allow_missing, a.expected_sha256)
    except (urllib.error.URLError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    print("Model installed." if installed else "No published model found (allowed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

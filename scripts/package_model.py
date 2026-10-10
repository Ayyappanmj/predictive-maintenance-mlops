"""Package the approved model into dist/model-bundle.tar.gz.
Usage: python scripts/package_model.py [--model-dir models] [--out dist/model-bundle.tar.gz]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_bundle import package_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="dist/model-bundle.tar.gz")
    a = ap.parse_args()
    print(f"Packaged: {package_bundle(a.model_dir, a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

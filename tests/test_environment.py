"""Phase 1 smoke tests: environment and config are sane."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "configs" / "model_config.yaml") as f:
        return yaml.safe_load(f)


def test_core_libraries_import():
    import numpy, pandas, sklearn  # noqa: F401,E401,I001


def test_config_has_required_sections():
    cfg = load_config()
    for key in ("seed", "data", "split", "risk", "promotion"):
        assert key in cfg


def test_target_not_in_features_and_leaky_columns_dropped():
    data = load_config()["data"]
    assert data["target"] not in data["numeric_features"]
    for leaky in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        assert leaky in data["drop_columns"]


def test_risk_thresholds_ordered():
    risk = load_config()["risk"]
    assert 0 < risk["medium"] < risk["high"] < 1

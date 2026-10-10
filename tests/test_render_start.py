"""Deployment launcher tests (no processes are started)."""

import sys

from scripts.render_start import build_command, model_url


def test_api_command_binds_all_interfaces_and_given_port():
    cmd = build_command("api", 10000)
    assert cmd[:3] == [sys.executable, "-m", "uvicorn"] and "app.api:app" in cmd
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--port") + 1] == "10000"


def test_dashboard_command_binds_all_interfaces_and_given_port():
    cmd = build_command("dashboard", 8501)
    assert "streamlit" in cmd and "app/dashboard.py" in cmd
    assert cmd[cmd.index("--server.address") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--server.port") + 1] == "8501"


def test_unknown_service_rejected():
    try:
        build_command("worker", 1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_model_url_resolution():
    assert model_url({}) is None
    both = {"MODEL_URL": "https://x/y.tar.gz", "MODEL_REPO": "a/b"}
    assert model_url(both) == "https://x/y.tar.gz"
    assert model_url({"MODEL_REPO": "a/b"}) == (
        "https://github.com/a/b/releases/download/model-latest/model-bundle.tar.gz"
    )
    assert model_url({"MODEL_REPO": "a/b", "MODEL_TAG": "model-1"}).endswith(
        "/download/model-1/model-bundle.tar.gz"
    )

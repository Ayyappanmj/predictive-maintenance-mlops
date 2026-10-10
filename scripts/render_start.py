"""Start command for Render (and Docker): fetch the approved model, then launch a service.

  python scripts/render_start.py api         -> uvicorn on 0.0.0.0:$PORT
  python scripts/render_start.py dashboard   -> streamlit on 0.0.0.0:$PORT

Environment:
  PORT        assigned by Render (default 8000 locally)
  MODEL_REPO  OWNER/REPO of the public GitHub repo whose release holds the model
  MODEL_TAG   release tag (default model-latest)
  MODEL_URL   full bundle URL (overrides MODEL_REPO/MODEL_TAG)
  MODEL_DIR   where the model lives (default models)

The local filesystem is temporary on Render's free tier, so the model is fetched on every start.
If fetching fails, the service still starts (the API reports `degraded`) and logs why.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_command(service: str, port: int) -> list[str]:
    if service == "api":
        return [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
    if service == "dashboard":
        return [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/dashboard.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
    raise ValueError(f"Unknown service '{service}' (use 'api' or 'dashboard')")


def model_url(env=os.environ) -> str | None:
    if env.get("MODEL_URL"):
        return env["MODEL_URL"]
    if env.get("MODEL_REPO"):
        tag = env.get("MODEL_TAG", "model-latest")
        return f"https://github.com/{env['MODEL_REPO']}/releases/download/{tag}/model-bundle.tar.gz"
    return None


def fetch_model() -> None:
    url = model_url()
    if not url:
        print(
            "render_start: MODEL_REPO/MODEL_URL not set; using any model already on disk.",
            flush=True,
        )
        return
    from src.model_bundle import fetch_bundle

    try:
        ok = fetch_bundle(url, os.getenv("MODEL_DIR", "models"), allow_missing=True)
        print(
            "render_start: model installed."
            if ok
            else "render_start: no published model found (404); starting without one.",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001  (never block startup on a fetch problem)
        print(f"render_start: WARNING could not fetch model: {e}", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    cmd = build_command(argv[1], int(os.getenv("PORT", "8000")))
    fetch_model()
    os.execvp(cmd[0], cmd)  # replace this process; signals go straight to the server
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

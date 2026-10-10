# Deploying to Render (free tier)

Two separate web services are defined in `render.yaml`: the **API** (FastAPI, `/docs`, `/health`)
and the **dashboard** (Streamlit). Nothing here has been deployed or tested by the author of this
guide: follow the verification steps below and record what you actually see.

## Free-tier facts to plan around (verify at https://render.com/docs/free)
- A free web service **spins down after 15 minutes without traffic** and takes about a minute to
  wake up (cold start). The first request after idle is slow.
- The filesystem is **temporary**: it is wiped on every restart, redeploy and spin-down. Nothing
  you train or save on the server survives. This project therefore fetches the approved model on
  every start (see below).
- Free instances have 512 MB RAM and a small CPU share. The API fits comfortably; the Streamlit
  dashboard is heavier and may be slow or run out of memory. If it crashes with an out-of-memory
  error, use the dashboard locally or on Streamlit Community Cloud instead.
- Free web services cannot receive private-network traffic, so the dashboard reaches the API by its
  public URL. Monthly free instance hours and build minutes are limited.

## How the model gets there
The local disk does not persist, so the approved model lives in a **GitHub Release** (created by the
retraining workflow's `publish` job) and `scripts/render_start.py` downloads it at startup:
`https://github.com/OWNER/REPO/releases/download/model-latest/model-bundle.tar.gz`.
The download is hash-verified. If no model has been published yet (404) or the download fails, the
service still starts: `/health` shows `degraded` and `/predict` returns 503 until a model exists.
The repository must be **public**. Publishing a new model does **not** redeploy anything: the new
model is picked up the next time a service starts (after idle spin-down, a manual restart, or a
deploy).

## Step by step
1. Complete Phase 9 and make sure a model release exists: Actions > "Scheduled retraining" >
   Run workflow with `publish = true`. Check Releases for `model-latest`.
2. In Render: **New > Blueprint**, connect your GitHub account and select the repository. Render
   reads `render.yaml` and shows two services.
3. When prompted, set `MODEL_REPO` to `YOUR-USER/predictive-maintenance-mlops` for both services.
   Leave `API_URL` blank for now.
4. Deploy. When the API is live, copy its URL (`https://predictive-maintenance-api-xxxx.onrender.com`),
   then set `API_URL` on the dashboard service (Environment tab) and redeploy it.
5. Manual alternative (no Blueprint): New > Web Service > repo; runtime Python 3;
   build `pip install -r requirements-api.txt` (dashboard: `requirements.txt`); start
   `python scripts/render_start.py api` (dashboard: `... dashboard`); health check path `/health`
   (dashboard: `/_stcore/health`); add the environment variables from the table.

| Variable | Service | Purpose |
|---|---|---|
| `PYTHON_VERSION` | both | `3.11.9` (set in the Blueprint) |
| `MODEL_REPO` | both | `OWNER/REPO` holding the model release |
| `MODEL_TAG` | both, optional | release tag, default `model-latest` |
| `MODEL_DIR` | both | default `models` |
| `API_URL` | dashboard | public URL of the API (for "FastAPI service" mode) |
| `LOG_LEVEL` | API | `INFO` |
| `RISK_THRESHOLD_MEDIUM`, `RISK_THRESHOLD_HIGH` | API, optional | override risk bands |

Both services bind `0.0.0.0` on the `PORT` Render assigns (handled in `scripts/render_start.py`).

## Verify (and only then fill in README URLs)
```bash
curl https://YOUR-API.onrender.com/health        # expect "status":"ok","model_loaded":true
curl https://YOUR-API.onrender.com/model-info
# open https://YOUR-API.onrender.com/docs
```
The first call after idle can take about a minute. Open the dashboard URL and make one prediction.

## Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `/health` says `degraded` | No model fetched. Check the service logs for `render_start:` lines; confirm the `model-latest` release exists, the repo is public, and `MODEL_REPO` is exact `OWNER/REPO`. Restart the service. |
| Build fails on `pip install` | Check the Python version (`PYTHON_VERSION`) and read the pip error; try `requirements-api.txt` for the API. |
| "No open ports detected" / deploy times out | The server must listen on `$PORT` on `0.0.0.0`; use the provided start command unchanged. |
| First request very slow, then fine | Free-tier cold start (about a minute). Expected. |
| Dashboard in "FastAPI service" mode fails | API is asleep or `API_URL` is wrong or has a trailing path. Wake the API first. |
| Dashboard killed or restarts repeatedly | Out of memory on 512 MB. Run it locally, or upgrade the plan. |
| Model changes don't appear | Services only fetch on start. Restart the service after publishing. |
| `/docs` missing | Wrong service URL (that is the dashboard) or the API failed to start; read the logs. |

## Docker (optional)
The `Dockerfile` builds the API image for local use or any Docker host (untested here):
```bash
docker build -t pm-api .
docker run -p 8000:8000 -e MODEL_REPO=OWNER/REPO pm-api
# or mount a trained model:  docker run -p 8000:8000 -v "$PWD/models:/app/models" pm-api
```

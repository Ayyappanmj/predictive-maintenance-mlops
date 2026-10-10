# API image (local use or any Docker host). Render uses the native Python runtime in render.yaml.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
# Model files are not baked in: mount ./models or set MODEL_REPO to fetch a published model.

EXPOSE 8000
CMD ["python", "scripts/render_start.py", "api"]

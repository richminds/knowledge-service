# syntax=docker/dockerfile:1

# ---- Build stage: compile wheels into a virtualenv (build tools stay out of the runtime) ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    # Keep 1 while ingestion job status lives in memory (see IngestionService); raise
    # once jobs move to a shared store.
    WEB_CONCURRENCY=1

# curl: container HEALTHCHECK (file types are detected from content in Python).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser config ./config
# Worker / scheduled jobs (ingestion_worker.py, reconcile.py) run from the same image.
COPY --chown=appuser:appuser scripts ./scripts

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# --proxy-headers: App Service terminates TLS in front of the container.
CMD ["sh", "-c", "exec uvicorn app.main:app --app-dir src --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*'"]

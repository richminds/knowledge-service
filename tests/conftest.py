"""Shared fixtures.

No test in this suite makes a real MongoDB or LLM Gateway call — the env
defaults below keep both settings singletons hermetic (real environment
variables outrank .env in pydantic-settings, so this also keeps the suite
deterministic on a developer machine with a fully populated .env). Settings
are module-level singletons built at import time, so these must be set
*before* any ``app``/``rag`` import happens.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_MONGO_URI", "")
os.environ.setdefault("RAG_AUTH_ENABLED", "false")
os.environ.setdefault("KNOWLEDGE_API_KEYS", "")
os.environ.setdefault("KNOWLEDGE_ENVIRONMENT", "test")
# Loopback + closed port: a request against this fails fast with "connection
# refused" instead of a slow DNS timeout, and no test relies on it succeeding.
os.environ.setdefault("RAG_GATEWAY_BASE_URL", "http://127.0.0.1:1")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def api():
    """A TestClient running the real app (real lifespan, real middleware).

    RAG_MONGO_URI is empty in the test environment, so the lifespan's index
    check is a no-op (see app/main.py::_ensure_mongo_indexes) and no real
    MongoDB connection is attempted merely by building the app.
    """
    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client

"""Auth middleware, exercised end-to-end against a real app instance.

Uses GET /v1/ingest/{job_id} against a nonexistent job — that route reaches
the in-memory job registry (no MongoDB / LLM Gateway needed), so a 404 proves
the request got PAST auth, and a 401 proves it did not.

Patches ``app.config.gateway_settings`` in place (``monkeypatch.setattr`` on
an *attribute* of the existing singleton) rather than swapping in a whole new
settings object. The latter looks equivalent but reliably left the patched
value visible to a *later, unrelated* test — a Starlette ``TestClient``
implementation detail unrelated to this service's own code (confirmed by
reproducing the same leak with a bare ``BaseHTTPMiddleware`` reading a
module-level global, independent of anything RAG- or auth-specific here).
Mutating the field in place avoids it and is the more standard monkeypatch
usage anyway.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _configured_client(monkeypatch, api_keys: str) -> TestClient:
    import app.config as config_mod

    monkeypatch.setattr(config_mod.gateway_settings, "api_keys", api_keys)

    from app.main import create_app

    return TestClient(create_app())


def test_missing_credentials_rejected_when_api_keys_configured(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key") as client:
        resp = client.get("/v1/ingest/does-not-exist")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"


def test_wrong_key_rejected(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key") as client:
        resp = client.get("/v1/ingest/does-not-exist", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401


def test_valid_api_key_reaches_the_route(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key") as client:
        resp = client.get("/v1/ingest/does-not-exist", headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 404  # past auth; job simply doesn't exist


def test_bearer_style_key_also_accepted(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key") as client:
        resp = client.get(
            "/v1/ingest/does-not-exist", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status_code == 404


def test_health_is_always_public_even_when_auth_configured(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key") as client:
        resp = client.get("/health/live")
        assert resp.status_code == 200


def test_admin_role_required_for_admin_routes(monkeypatch):
    with _configured_client(monkeypatch, "test-app:secret-key:user") as client:
        resp = client.get("/v1/config", headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 403


def test_open_mode_when_nothing_configured(api):
    # conftest.py leaves KNOWLEDGE_API_KEYS="" and RAG_AUTH_ENABLED=false.
    resp = api.get("/v1/ingest/does-not-exist")
    assert resp.status_code == 404  # open mode: no credential required

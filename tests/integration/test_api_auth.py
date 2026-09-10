"""Auth middleware, exercised end-to-end against a real app instance.

This service validates nothing itself. The API gateway authenticates the
caller against auth-service, strips whatever identity headers the caller sent,
and injects verified ones; this middleware reads those. So these tests send
``X-User-ID`` / ``X-Account-ID`` / ``X-Is-Admin`` directly — which is exactly
what the gateway does, and exactly what an attacker could do if the service
were reachable without the gateway in front (see app/middleware/auth.py).

Static service API keys (``KNOWLEDGE_API_KEYS``) and local JWT validation
(``RAG_JWT_SECRET``) were both removed — see ``app/middleware/auth.py`` for
why a second, weaker credential path was worth deleting rather than keeping.

Uses GET /v1/ingest/{job_id} against a nonexistent job — that route reaches
the in-memory job registry (no MongoDB / LLM Gateway needed), so a 404 proves
the request got PAST auth, and a 401 proves it did not.

Patches the settings singletons in place (``monkeypatch.setattr`` on an
*attribute* of the existing object) rather than swapping in a whole new
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


def _enforced_client(monkeypatch) -> TestClient:
    """A client against an app with gateway identity enforced."""
    import rag.config as rag_config

    monkeypatch.setattr(rag_config.settings, "auth_enabled", True)

    from app.main import create_app

    return TestClient(create_app())


def _identity(user_id: str = "USR-1", account_id: str = "", admin: bool = False) -> dict:
    """The headers the gateway injects for a verified caller."""
    headers = {"X-User-ID": user_id, "X-Authenticated-Via": "api-gateway"}
    if account_id:
        headers["X-Account-ID"] = account_id
    if admin:
        headers["X-Is-Admin"] = "true"
    return headers


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def test_missing_identity_rejected_when_auth_enabled(monkeypatch):
    with _enforced_client(monkeypatch) as client:
        resp = client.get("/v1/ingest/does-not-exist")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"


def test_a_bearer_token_alone_does_not_authenticate(monkeypatch):
    """The token still travels — this service forwards it on its own outbound
    calls — but it is no longer what identifies the caller. Only the gateway's
    injected headers are, so a raw token without them is not a credential
    here."""
    with _enforced_client(monkeypatch) as client:
        resp = client.get(
            "/v1/ingest/does-not-exist",
            headers={"Authorization": "Bearer whatever.this.is"},
        )
        assert resp.status_code == 401


def test_gateway_identity_reaches_the_route(monkeypatch):
    with _enforced_client(monkeypatch) as client:
        resp = client.get("/v1/ingest/does-not-exist", headers=_identity())
        assert resp.status_code == 404  # past auth; the job simply doesn't exist


def test_an_api_key_header_no_longer_authenticates_anything(monkeypatch):
    """The mechanism is gone, not merely unconfigured: a caller still sending
    X-API-Key gets a 401 rather than quietly reaching the route."""
    with _enforced_client(monkeypatch) as client:
        resp = client.get(
            "/v1/ingest/does-not-exist", headers={"X-API-Key": "secret-key"}
        )
        assert resp.status_code == 401


def test_health_is_always_public_even_when_auth_configured(monkeypatch):
    with _enforced_client(monkeypatch) as client:
        assert client.get("/health/live").status_code == 200


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

def test_admin_routes_reject_an_ordinary_caller(monkeypatch):
    with _enforced_client(monkeypatch) as client:
        resp = client.get("/v1/config", headers=_identity())
        assert resp.status_code == 403


def test_admin_routes_reject_a_falsey_is_admin_header(monkeypatch):
    """Anything other than "true" means "user", never "admin" — so a header
    the gateway did not set, or set to something else, cannot open an admin
    route."""
    with _enforced_client(monkeypatch) as client:
        headers = _identity() | {"X-Is-Admin": "false"}
        assert client.get("/v1/config", headers=headers).status_code == 403

        headers = _identity() | {"X-Is-Admin": "1"}
        assert client.get("/v1/config", headers=headers).status_code == 403


def test_admin_routes_accept_the_admin_header(monkeypatch):
    """The gateway sets X-Is-Admin from auth-service's is_admin, which is
    derived from the account the token is scoped to. Without it these routes
    are unreachable for everyone, which is what removing API keys would
    otherwise have caused."""
    with _enforced_client(monkeypatch) as client:
        resp = client.get("/v1/config", headers=_identity(admin=True))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Open mode
# ---------------------------------------------------------------------------

def test_open_mode_when_auth_is_disabled(api):
    """conftest.py leaves RAG_AUTH_ENABLED=false — the local-development
    default, and now the only remaining way for this service to be open."""
    resp = api.get("/v1/ingest/does-not-exist")
    assert resp.status_code == 404


def test_open_mode_does_not_grant_admin(api):
    """Regression: disabling auth used to set role="admin" for every caller,
    which made /v1/config and /v1/stats world-readable on any deployment that
    had not turned auth on yet. Disabled auth must mean *less* access, not
    total access."""
    assert api.get("/v1/config").status_code == 403

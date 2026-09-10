"""Auth middleware, exercised end-to-end against a real app instance.

A JWT is the only credential this service accepts. Static service API keys
(``KNOWLEDGE_API_KEYS``) were removed once every caller began arriving through
the API Gateway with a token auth-service had already verified — see
``app/middleware/auth.py`` for why a second, weaker credential path was worth
deleting rather than keeping.

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

SECRET = "test-secret-that-is-long-enough-for-hs256"


def _jwt_client(monkeypatch) -> TestClient:
    """A client against an app with JWT enforcement switched on."""
    import rag.config as rag_config

    monkeypatch.setattr(rag_config.settings, "auth_enabled", True)
    monkeypatch.setattr(rag_config.settings, "jwt_secret", SECRET)

    from app.main import create_app

    return TestClient(create_app())


def _token(**claims) -> str:
    """Mint a token this service will accept, with the given extra claims."""
    from rag.auth import JWTValidator

    return JWTValidator(secret=SECRET).create_token(
        claims.pop("subject", "USR-1"), **claims
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def test_missing_credentials_rejected_when_auth_enabled(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        resp = client.get("/v1/ingest/does-not-exist")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"


def test_a_garbage_token_is_rejected(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        resp = client.get("/v1/ingest/does-not-exist", headers=_bearer("not-a-jwt"))
        assert resp.status_code == 401


def test_a_token_signed_with_another_secret_is_rejected(monkeypatch):
    from rag.auth import JWTValidator

    forged = JWTValidator(secret="a-different-secret-entirely").create_token("USR-1")
    with _jwt_client(monkeypatch) as client:
        assert client.get(
            "/v1/ingest/does-not-exist", headers=_bearer(forged)
        ).status_code == 401


def test_an_expired_token_is_rejected(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        expired = _token(ttl_seconds=-60)
        assert client.get(
            "/v1/ingest/does-not-exist", headers=_bearer(expired)
        ).status_code == 401


def test_a_valid_token_reaches_the_route(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        resp = client.get("/v1/ingest/does-not-exist", headers=_bearer(_token()))
        assert resp.status_code == 404  # past auth; the job simply doesn't exist


def test_an_api_key_header_no_longer_authenticates_anything(monkeypatch):
    """The mechanism is gone, not merely unconfigured: a caller still sending
    X-API-Key gets a 401 rather than quietly reaching the route."""
    with _jwt_client(monkeypatch) as client:
        resp = client.get(
            "/v1/ingest/does-not-exist", headers={"X-API-Key": "secret-key"}
        )
        assert resp.status_code == 401


def test_health_is_always_public_even_when_auth_configured(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        assert client.get("/health/live").status_code == 200


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

def test_admin_routes_reject_an_ordinary_role(monkeypatch):
    with _jwt_client(monkeypatch) as client:
        resp = client.get("/v1/config", headers=_bearer(_token(role="user")))
        assert resp.status_code == 403


def test_admin_routes_reject_a_token_with_no_role_claim(monkeypatch):
    """Absent means "user", never "admin" — a token minted before the claim
    existed must not open an admin route."""
    with _jwt_client(monkeypatch) as client:
        assert client.get(
            "/v1/config", headers=_bearer(_token())
        ).status_code == 403


def test_admin_routes_accept_the_admin_role(monkeypatch):
    """auth-service mints role="admin" for a session scoped to the admin app
    account. Without that claim these routes are unreachable for everyone,
    which is what removing API keys would otherwise have caused."""
    with _jwt_client(monkeypatch) as client:
        resp = client.get("/v1/config", headers=_bearer(_token(role="admin")))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Open mode
# ---------------------------------------------------------------------------

def test_open_mode_when_auth_is_disabled(api):
    """conftest.py leaves RAG_AUTH_ENABLED=false — the local-development
    default, and now the only remaining way for this service to be open."""
    resp = api.get("/v1/ingest/does-not-exist")
    assert resp.status_code == 404

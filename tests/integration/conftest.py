"""Shared fixtures for HTTP-level tests."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core.config import get_settings

TENANT = "11111111-1111-1111-1111-111111111111"
AUDIENCE = "api://rag-agent"


@pytest.fixture
def entra(monkeypatch):
    """Entra auth with a locally generated signing key instead of the real JWKS endpoint."""
    from app.core import auth

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class Jwks:
        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": key.public_key()})()

    monkeypatch.setattr(auth, "_jwks_client", lambda tenant: Jwks())
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_mode", "entra")
    monkeypatch.setattr(settings, "entra_tenant_id", TENANT)
    monkeypatch.setattr(settings, "entra_api_audience", [AUDIENCE])

    def token(groups=(), roles=(), audience=AUDIENCE, expires_in=300, oid="user-1"):
        now = int(time.time())
        claims = {
            "oid": oid,
            "preferred_username": "u@corp.org",
            "groups": list(groups),
            "roles": list(roles),
            "tid": TENANT,
            "aud": audience,
            "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
            "iat": now,
            "exp": now + expires_in,
        }
        return {"Authorization": f"Bearer {jwt.encode(claims, key, algorithm='RS256')}"}

    return token

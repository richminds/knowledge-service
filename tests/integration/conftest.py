"""Shared fixtures for HTTP-level tests."""

import pytest

from app.core.config import get_settings


@pytest.fixture
def gateway(monkeypatch):
    """AUTH_MODE=gateway; returns a builder for the identity headers the API gateway injects."""
    monkeypatch.setattr(get_settings(), "auth_mode", "gateway")

    def headers(user_id="user-1", account_id="", is_admin=False, email="u@corp.org", name=""):
        result = {"X-User-ID": user_id, "X-User-Email": email}
        if name:
            result["X-User-Name"] = name
        if account_id:
            result["X-Account-ID"] = account_id
        if is_admin:  # the gateway only ever sends the header when true
            result["X-Is-Admin"] = "true"
        return result

    return headers

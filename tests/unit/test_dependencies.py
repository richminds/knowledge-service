from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.dependencies import require_admin, resolve_org_id, resolve_user_id


def _request(**state) -> SimpleNamespace:
    """Minimal stand-in for a Starlette Request — resolve_user_id/resolve_org_id
    only ever touch ``request.state``."""
    return SimpleNamespace(state=SimpleNamespace(**state))


def test_require_admin_allows_admin_role():
    assert require_admin(role="admin") == "admin"


def test_require_admin_rejects_non_admin_role():
    with pytest.raises(HTTPException) as exc_info:
        require_admin(role="user")
    assert exc_info.value.status_code == 403


# ── resolve_user_id: trusted caller (API key / auth disabled) ──────────────
# The Portless-style pattern: a trusted service asserts an end-user ID on
# behalf of one of its own users.

def test_resolve_user_id_prefers_explicit_value_for_trusted_caller():
    request = _request(principal="portless-backend", auth_method="api_key")
    assert resolve_user_id("alice", request) == "alice"


def test_resolve_user_id_falls_back_to_principal_when_absent():
    request = _request(principal="portless-backend", auth_method="api_key")
    assert resolve_user_id(None, request) == "portless-backend"


def test_resolve_user_id_falls_back_to_principal_when_blank():
    request = _request(principal="portless-backend", auth_method="api_key")
    assert resolve_user_id("   ", request) == "portless-backend"


def test_resolve_user_id_trusts_explicit_value_when_auth_disabled():
    # No auth_method attribute at all — same as AuthMiddleware's open-mode branch.
    request = _request(principal="anonymous")
    assert resolve_user_id("alice", request) == "alice"


# ── resolve_user_id: JWT-authenticated end user — identity is the token's sub

def test_resolve_user_id_jwt_uses_token_subject_when_omitted():
    request = _request(principal="USR-1", auth_method="jwt")
    assert resolve_user_id(None, request) == "USR-1"


def test_resolve_user_id_jwt_allows_matching_explicit_value():
    request = _request(principal="USR-1", auth_method="jwt")
    assert resolve_user_id("USR-1", request) == "USR-1"


def test_resolve_user_id_jwt_rejects_mismatched_explicit_value():
    request = _request(principal="USR-1", auth_method="jwt")
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id("USR-2", request)
    assert exc_info.value.status_code == 403


# ── resolve_org_id: trusted caller (API key / auth disabled) ───────────────


def test_resolve_org_id_defaults_to_empty_for_trusted_caller():
    request = _request(auth_method="api_key")
    assert resolve_org_id(None, request) == ""


# ── resolve_org_id: JWT-authenticated end user — org_id is the hard tenant
# boundary, so it comes ONLY from the verified token's claim.





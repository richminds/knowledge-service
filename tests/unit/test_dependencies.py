import pytest
from fastapi import HTTPException

from app.dependencies import require_admin, resolve_user_id


def test_require_admin_allows_admin_role():
    assert require_admin(role="admin") == "admin"


def test_require_admin_rejects_non_admin_role():
    with pytest.raises(HTTPException) as exc_info:
        require_admin(role="user")
    assert exc_info.value.status_code == 403


def test_resolve_user_id_prefers_explicit_value():
    assert resolve_user_id("alice", principal="portless-backend") == "alice"


def test_resolve_user_id_falls_back_to_principal_when_absent():
    assert resolve_user_id(None, principal="portless-backend") == "portless-backend"


def test_resolve_user_id_falls_back_to_principal_when_blank():
    assert resolve_user_id("   ", principal="portless-backend") == "portless-backend"

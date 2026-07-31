import pytest
from fastapi import HTTPException

from app.dependencies import require_admin


def test_require_admin_allows_admin_role():
    assert require_admin(role="admin") == "admin"


def test_require_admin_rejects_non_admin_role():
    with pytest.raises(HTTPException) as exc_info:
        require_admin(role="user")
    assert exc_info.value.status_code == 403

"""Caller identity, read off the headers the API gateway injects, and the principals used for
ACL filtering (C-048, C-154).

This service validates no tokens and holds no signing key. Every caller reaches it through
the API gateway, which authenticates the request against auth-service (``GET /auth/me``),
**strips** any identity headers the caller tried to send, and injects verified ones. Same
contract as makemerich-backend (app/auth/dependencies.py) and llm-gateway:

  ``X-User-ID``      auth-service's ``user_id`` (``USR-<hex>``)
  ``X-User-Email``   the record's email
  ``X-User-Name``    display name
  ``X-Account-ID``   the app account the token is scoped to (omitted when there is none)
  ``X-Is-Admin``     ``true`` for admins (only ever sent when true)

The caller's bearer token is forwarded too; it is not validated here, only carried on to
the LLM gateway (app/core/caller_context.py).

AUTH_MODE:
  gateway - the headers above are required; no ``X-User-ID`` means 401
  none    - no authentication; ACLs are NOT enforced (local development only)

**The trade this makes.** Trusting a header is only sound while the caller cannot set it.
The gateway's unconditional strip guarantees that for traffic through the gateway, and
nothing else does: restrict this service to the gateway at the network level.
"""

from dataclasses import dataclass, field, replace
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.types import PUBLIC

logger = get_logger(__name__)


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str = ""
    email: str = ""
    account_id: str = ""
    is_admin: bool = False
    auth_method: str = "none"  # none | gateway
    extra_principals: tuple[str, ...] = field(default_factory=tuple)
    enforce_acl: bool = True

    @property
    def principals(self) -> frozenset[str] | None:
        """ACL principals for RetrievalFilters; None disables ACL enforcement."""
        if not self.enforce_acl:
            return None
        account = (self.account_id,) if self.account_id else ()
        return frozenset({PUBLIC, self.user_id, *account, *self.extra_principals})

    @property
    def rate_limit_key(self) -> str:
        return f"{self.auth_method}:{self.user_id}"


ANONYMOUS = UserContext(user_id="anonymous", auth_method="none", enforce_acl=False)

CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def identity_from_headers(request: Request) -> UserContext | None:
    """The identity the gateway asserted, or None when there is none.

    Read fresh from headers on every request: the gateway re-derives them per request, and
    this service never holds an opinion about identity that outlives one call.
    """
    user_id = (request.headers.get("x-user-id") or "").strip()
    if not user_id:
        return None
    email = (request.headers.get("x-user-email") or "").strip().lower()
    return UserContext(
        user_id=user_id,
        name=(request.headers.get("x-user-name") or "").strip() or email,
        email=email,
        account_id=(request.headers.get("x-account-id") or "").strip(),
        is_admin=(request.headers.get("x-is-admin") or "").strip().lower() == "true",
        auth_method="gateway",
    )


async def get_current_user(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> UserContext:
    if settings.auth_mode == "none":
        user = ANONYMOUS
    else:
        user = identity_from_headers(request)
        if user is None:
            raise CREDENTIALS_EXCEPTION
    if not settings.acl_enforcement and user.enforce_acl:
        user = replace(user, enforce_acl=False)
    structlog.contextvars.bind_contextvars(user_id=user.user_id, auth_method=user.auth_method)
    request.state.user = user
    return user


CurrentUser = Annotated[UserContext, Depends(get_current_user)]


def _require_admin(user: UserContext) -> None:
    # Local (AUTH_MODE=none) callers are trusted; gateway callers need X-Is-Admin.
    if user.auth_method == "gateway" and not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Requires an admin account")


async def require_ingestion_role(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> UserContext:
    """Ingestion/deletion endpoints: admins only when INGESTION_REQUIRES_ADMIN is on."""
    if settings.ingestion_requires_admin:
        _require_admin(user)
    return user


IngestionUser = Annotated[UserContext, Depends(require_ingestion_role)]


async def require_admin_role(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> UserContext:
    """Admin RAG studio: admins only when ADMIN_REQUIRES_ADMIN is on (the default)."""
    if settings.admin_requires_admin:
        _require_admin(user)
    return user


AdminUser = Annotated[UserContext, Depends(require_admin_role)]

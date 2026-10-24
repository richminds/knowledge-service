"""Caller authentication and the principals used for ACL filtering (C-048, C-154).

AUTH_MODE:
  none     - no authentication; ACLs are NOT enforced (local development only)
  api_key  - shared X-API-Key for service-to-service calls; the caller gets
             API_KEY_PRINCIPALS (default "*": documents shared with everyone)
  entra    - Entra ID bearer tokens (user sign-in); principals = user object id + group ids
  entra+api_key - either of the two

Entra setup (placeholders in config): register an API app, expose a scope, set
ENTRA_TENANT_ID and ENTRA_API_AUDIENCE, and enable the `groups` claim (Token configuration
-> Add groups claim -> Security groups). Users in > 200 groups get a group-overage claim
instead; resolving it needs Microsoft Graph (TODO: ENTRA_GROUP_OVERAGE=graph).
"""

import asyncio
import hashlib
import secrets
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Annotated

import jwt
import structlog
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.types import PUBLIC

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str = ""
    groups: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    tenant_id: str = ""
    auth_method: str = "none"  # none | api_key | entra
    extra_principals: tuple[str, ...] = field(default_factory=tuple)
    enforce_acl: bool = True

    @property
    def principals(self) -> frozenset[str] | None:
        """ACL principals for RetrievalFilters; None disables ACL enforcement."""
        if not self.enforce_acl:
            return None
        return frozenset({PUBLIC, self.user_id, *self.groups, *self.extra_principals})

    @property
    def rate_limit_key(self) -> str:
        return f"{self.auth_method}:{self.user_id}"


ANONYMOUS = UserContext(user_id="anonymous", auth_method="none", enforce_acl=False)


@lru_cache(maxsize=4)
def _jwks_client(tenant_id: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
        cache_keys=True,
        lifespan=3600,
    )


def _decode_entra_token(token: str, settings: Settings) -> dict:
    tenant = settings.entra_tenant_id
    signing_key = _jwks_client(tenant).get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=[a for a in settings.entra_api_audience if a],
        # v2 and v1 token issuers
        issuer=[
            f"https://login.microsoftonline.com/{tenant}/v2.0",
            f"https://sts.windows.net/{tenant}/",
        ],
        options={"require": ["exp", "iat", "aud", "iss"]},
    )


async def _entra_user(token: str, settings: Settings) -> UserContext:
    try:
        # PyJWKClient fetches keys over the network (cached); keep it off the event loop.
        claims = await asyncio.to_thread(_decode_entra_token, token, settings)
    except jwt.PyJWTError as exc:
        logger.info("auth_rejected", reason=str(exc))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid bearer token") from exc
    if "_claim_names" in claims and "groups" in claims.get("_claim_names", {}):
        logger.warning("entra_group_overage", user=claims.get("oid"))
    return UserContext(
        user_id=claims.get("oid") or claims.get("sub", ""),
        name=claims.get("preferred_username") or claims.get("name", ""),
        groups=tuple(claims.get("groups") or ()),
        roles=tuple(claims.get("roles") or ()),
        tenant_id=claims.get("tid", ""),
        auth_method="entra",
    )


def _api_key_user(settings: Settings, api_key: str) -> UserContext:
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:12]
    return UserContext(
        user_id=f"service:{fingerprint}",
        name="api-key",
        auth_method="api_key",
        extra_principals=tuple(settings.api_key_principals),
    )


async def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    api_key: Annotated[str | None, Security(_api_key_header)] = None,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)] = None,
) -> UserContext:
    mode = settings.auth_mode
    user: UserContext | None = None

    if mode == "none":
        user = ANONYMOUS
    if user is None and "entra" in mode and bearer is not None:
        if not settings.entra_tenant_id:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Entra auth not configured")
        user = await _entra_user(bearer.credentials, settings)
    if user is None and "api_key" in mode and api_key:
        expected = settings.api_key.get_secret_value()
        if expected and secrets.compare_digest(api_key, expected):
            user = _api_key_user(settings, api_key)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentication required",
            headers={"WWW-Authenticate": "Bearer" if "entra" in mode else "ApiKey"},
        )
    if not settings.acl_enforcement and user.enforce_acl:
        user = replace(user, enforce_acl=False)
    structlog.contextvars.bind_contextvars(user_id=user.user_id, auth_method=user.auth_method)
    request.state.user = user
    return user


CurrentUser = Annotated[UserContext, Depends(get_current_user)]


async def require_ingestion_role(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> UserContext:
    """Ingestion/deletion endpoints: Entra users need INGESTION_REQUIRED_ROLE (an app role,
    e.g. "Ingestion.Write") when it is set; API-key and local callers are trusted."""
    role = settings.ingestion_required_role
    if role and user.auth_method == "entra" and role not in user.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires the '{role}' role")
    return user


IngestionUser = Annotated[UserContext, Depends(require_ingestion_role)]


async def require_admin_role(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> UserContext:
    """Admin RAG studio: Entra users need ADMIN_REQUIRED_ROLE (case-insensitive, as the UI
    checks it); API-key and local callers are trusted, like ingestion."""
    role = settings.admin_required_role.strip().lower()
    if role and user.auth_method == "entra" and role not in {r.lower() for r in user.roles}:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, f"Requires the '{settings.admin_required_role}' role"
        )
    return user


AdminUser = Annotated[UserContext, Depends(require_admin_role)]

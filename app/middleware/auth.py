"""Authentication + role-based authorization middleware.

Two authentication mechanisms, checked in order, so a deployment can pick
whichever fits:

1. **Static service API keys** (``KNOWLEDGE_API_KEYS``) — the pragmatic choice
   for service-to-service traffic (e.g. Portless, or any other application
   calling this service). Sent as ``X-API-Key: <key>`` or
   ``Authorization: Bearer <key>``. The configured name becomes the request's
   principal. Format is ``name:key`` or ``name:key:role`` — role defaults to
   ``"user"``.
2. **JWT** (``RAG_AUTH_ENABLED`` + ``RAG_JWT_SECRET``) — validated by
   ``rag.auth.JWTValidator``; the token's ``sub`` becomes the principal, its
   ``role`` claim becomes the role (default ``"user"``), and its ``org_id``
   and ``account_id`` claims become ``request.state.org_id`` /
   ``request.state.account_id`` — the verified boundaries that
   ``app/dependencies.py``'s ``resolve_org_id``/``resolve_account_id`` use
   instead of trusting client-supplied fields.

Either way, the resolved role lands on ``request.state.role`` — see
``app/dependencies.py``'s ``require_admin`` for how routes gate on it.

With neither mechanism configured the API is **open** and every request is
"anonymous" with role "admin" — fine for local dev and the test suite, wrong
for anything reachable off localhost — ``main.py`` logs a startup warning.
Configuring EITHER mechanism turns on enforcement everywhere, admin routes
included. Ported from the LLM Gateway's ``app/middleware/auth.py``.

Health and docs paths are always public so orchestrator probes work before
any credential is provisioned.
"""
from __future__ import annotations

import hmac
import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.auth import InvalidTokenError, JWTValidator
from rag.config import settings as rag_settings

from ..config import gateway_settings

logger = logging.getLogger(__name__)

# Prefixes that never require a credential.
PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _is_public(path: str) -> bool:
    return path == "/" or path.startswith(PUBLIC_PATH_PREFIXES)


def _match_api_key(
    presented: str, known: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """Constant-time-ish lookup of a presented key.

    ``dict`` lookup would be a timing oracle on key contents, so compare against
    every configured key with ``compare_digest`` and never short-circuit.
    Returns (principal_name, role) or None.
    """
    matched: tuple[str, str] | None = None
    for key, name_role in known.items():
        if hmac.compare_digest(presented, key):
            matched = name_role
    return matched


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, validator: JWTValidator | None = None) -> None:
        super().__init__(app)
        # Left as None so the validator is built from *current* settings on each
        # use. Binding one at construction time would freeze the secret at
        # import order, which breaks any deployment that enables JWT after the
        # app object exists.
        self._validator = validator

    def _jwt_validator(self) -> JWTValidator:
        return self._validator or JWTValidator()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if _is_public(request.url.path):
            request.state.principal = "public"
            request.state.role = "admin"  # public paths touch no protected data
            return await call_next(request)

        api_keys = gateway_settings.parsed_api_keys()
        jwt_enabled = rag_settings.auth_enabled

        # ── open mode: nothing configured ──────────────────────────────────
        if not api_keys and not jwt_enabled:
            request.state.principal = "anonymous"
            request.state.role = "admin"
            return await call_next(request)

        header_key = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        bearer = (
            authorization[7:].strip()
            if authorization[:7].lower() == "bearer "
            else ""
        )

        # ── 1. static API key ──────────────────────────────────────────────
        if api_keys:
            for candidate in (header_key, bearer):
                if not candidate:
                    continue
                matched = _match_api_key(candidate, api_keys)
                if matched:
                    name, role = matched
                    request.state.principal = name
                    request.state.role = role
                    request.state.auth_method = "api_key"
                    return await call_next(request)

        # ── 2. JWT ─────────────────────────────────────────────────────────
        if jwt_enabled and bearer:
            try:
                claims = self._jwt_validator().validate(bearer)
            except InvalidTokenError as exc:
                logger.warning("JWT rejected on %s: %s", request.url.path, exc)
                return _unauthorized(str(exc))
            request.state.principal = claims.sub
            request.state.role = claims.extra.get("role") or "user"
            # The hard tenant-isolation boundary (see rag/authorization.py) —
            # taken ONLY from the verified token from here on, never from a
            # request body/query field. See app/dependencies.py::resolve_org_id.
            request.state.org_id = claims.extra.get("org_id") or ""
            # The application (auth-service app account) this token is scoped
            # to — the second isolation boundary, enforced exactly like
            # org_id. See app/dependencies.py::resolve_account_id.
            request.state.account_id = claims.extra.get("account_id") or ""
            request.state.auth_method = "jwt"
            return await call_next(request)

        return _unauthorized(
            "Missing or invalid credentials. Send X-API-Key: <key>"
            + (" or Authorization: Bearer <jwt>." if jwt_enabled else ".")
        )


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"code": "unauthorized", "message": detail}},
        headers={"WWW-Authenticate": "Bearer"},
    )

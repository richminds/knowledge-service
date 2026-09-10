"""Authentication + role-based authorization — one credential, one path.

Every caller reaches this service through the API Gateway, which authenticates
the request against auth-service before routing it and forwards the caller's
token untouched. A token minted by auth-service is therefore the only
credential this service accepts, and ``RAG_AUTH_ENABLED`` + ``RAG_JWT_SECRET``
is the only way to configure authentication.

**Static service API keys used to be accepted too, and are gone.**
``KNOWLEDGE_API_KEYS`` dated from when applications called this service
directly. A static key is a bearer credential with no expiry, no subject, no
account scope and no revocation — and the shape of that mattered here, because
``app/dependencies.py``'s ``resolve_account_id`` and ``resolve_user_id``
deliberately trust a *request body* field for API-key callers, having no
verified identity to check it against. Every isolation boundary in this service
was therefore optional for anyone holding a key. Something that needs to call
this service with no end-user behind it mints a JWT against the same shared
secret (``scripts/mint_token.py``): the same trust root, with a subject, an
expiry and an audience attached.

The verified token supplies:

  * ``sub`` → ``request.state.principal``
  * ``role`` → ``request.state.role``, defaulting to "user"; only "admin"
    reaches the admin routes (``app/dependencies.py``'s ``require_admin``).
    auth-service derives it from the account the token is scoped to.
  * ``account_id`` → ``request.state.account_id``, the hard isolation boundary
    ``resolve_account_id`` enforces instead of trusting a client-supplied
    field.

With ``RAG_AUTH_ENABLED`` false the API is **open** and every request is
"anonymous" with role "admin" — the local-development and test default, and
wrong for anything reachable off localhost. ``main.py`` logs a startup warning,
and this is now the *only* thing standing between a deployment and an open
door: with API keys removed there is no second mechanism that might happen to
be configured.

Health and docs paths are always public so orchestrator probes work before any
credential is provisioned.
"""
from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.auth import InvalidTokenError, JWTValidator
from rag.caller_context import caller_token_scope
from rag.config import settings as rag_settings

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

    async def _proceed(self, request, call_next, caller_token=""):
        """Run the rest of the stack with the caller's token bound.

        Every LLM and embedding call this service makes goes out through the
        API Gateway carrying this token, so that the gateway can meter and
        budget the traffic against the user who caused it — see
        rag/llm_gateway_client.py.
        """
        with caller_token_scope(caller_token):
            return await call_next(request)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if _is_public(request.url.path):
            request.state.principal = "public"
            request.state.role = "admin"  # public paths touch no protected data
            return await call_next(request)

        authorization = request.headers.get("authorization", "")
        bearer = (
            authorization[7:].strip()
            if authorization[:7].lower() == "bearer "
            else ""
        )

        # ── open mode: no JWT secret configured ────────────────────────────
        if not rag_settings.auth_enabled:
            request.state.principal = "anonymous"
            request.state.role = "admin"
            # Nothing was verified here, but the token still travels: the
            # gateway in front already checked it, and it is what identifies
            # the user on this service's own outbound calls.
            return await self._proceed(request, call_next, bearer)

        if not bearer:
            return _unauthorized(
                "This endpoint requires authentication. Send "
                "'Authorization: Bearer <jwt>' — obtain a token by signing in "
                "through the auth service behind the API gateway."
            )

        try:
            claims = self._jwt_validator().validate(bearer)
        except InvalidTokenError as exc:
            logger.warning("JWT rejected on %s: %s", request.url.path, exc)
            return _unauthorized(str(exc))

        request.state.principal = claims.sub
        request.state.role = claims.extra.get("role") or "user"
        # The application (auth-service app account) this token is scoped to —
        # the isolation boundary, taken ONLY from the verified token and never
        # from a request body/query field. See
        # app/dependencies.py::resolve_account_id.
        request.state.account_id = claims.extra.get("account_id") or ""
        request.state.auth_method = "jwt"
        return await self._proceed(request, call_next, bearer)


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"code": "unauthorized", "message": detail}},
        headers={"WWW-Authenticate": "Bearer"},
    )

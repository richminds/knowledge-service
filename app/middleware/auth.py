"""Authentication + role-based authorization — identity comes from the gateway.

This service does not validate tokens and holds no signing key. Every caller
reaches it through the API gateway, which authenticates the request against
auth-service (``GET /auth/me``), **strips** any identity headers the caller
tried to send, and injects verified ones. This middleware reads those:

  * ``X-User-ID``          → ``request.state.principal``
  * ``X-Is-Admin``         → ``request.state.role`` ("admin" when "true", else
    "user"); only "admin" reaches the admin routes (``app/dependencies.py``'s
    ``require_admin``). auth-service derives it from the account the token is
    scoped to.
  * ``X-Account-ID``       → ``request.state.account_id``, the hard isolation
    boundary ``resolve_account_id`` enforces instead of trusting a
    client-supplied field.

**Why not verify the JWT here.** It used to. That required ``RAG_JWT_SECRET``
to hold the same signing key as auth-service and llm-gateway — three copies of
one secret, kept in sync by hand, which had already drifted apart. Local
verification also cannot see auth-service's revocation list, so a logged-out
token kept working here until it expired. The gateway asks auth-service, which
checks signature, expiry *and* revocation, and caches the answer.

**The trade this makes.** Trusting a header is only sound while the caller
cannot set it. The gateway's unconditional strip guarantees that for traffic
through the gateway — and nothing else does. A deployment that leaves this
service reachable on its own URL lets anyone send ``X-Account-ID`` and read
another account's data. Restrict it to the gateway (on Vercel: Deployment
Protection) — the startup warning below fires when ``RAG_AUTH_ENABLED`` is
false, but nothing can detect direct reachability from inside the process.

**Static service API keys are gone too** (``KNOWLEDGE_API_KEYS``). A static key
was a bearer credential with no expiry, no subject, no account scope and no
revocation. Something that needs to call this service with no end user behind
it goes through the gateway like everything else.

Health and docs paths are always public so orchestrator probes work before any
credential is provisioned.
"""
from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

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


def _identity(request: Request) -> tuple[str, str, bool]:
    """(user_id, account_id, is_admin) as the gateway asserted them.

    Read fresh from headers on every request rather than cached anywhere: the
    gateway re-derives them per request from its introspection cache, and this
    service should never hold an opinion about identity that outlives one call.
    """
    user_id = (request.headers.get("x-user-id") or "").strip()
    account_id = (request.headers.get("x-account-id") or "").strip()
    is_admin = (request.headers.get("x-is-admin") or "").strip().lower() == "true"
    return user_id, account_id, is_admin


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)

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
        user_id, account_id, is_admin = _identity(request)

        # ── open mode: no gateway in front ─────────────────────────────────
        if not rag_settings.auth_enabled:
            request.state.principal = user_id or "anonymous"
            # "user", NOT "admin". This branch used to grant admin to every
            # caller, which made /v1/stats and /v1/config world-readable on any
            # deployment that had not turned auth on yet — the opposite of what
            # a disabled-auth default should do.
            request.state.role = "admin" if is_admin else "user"
            # Nothing was verified, so account_id stays unset and
            # resolve_account_id falls back to the request body — see its
            # docstring. Local development only.
            return await self._proceed(request, call_next, bearer)

        # ── enforced: the gateway must have said who this is ───────────────
        if not user_id:
            return _unauthorized(
                "This endpoint is reachable only through the API gateway, which "
                "supplies the caller's verified identity. Sign in through the "
                "auth service and call the gateway rather than this service "
                "directly."
            )

        request.state.principal = user_id
        request.state.role = "admin" if is_admin else "user"
        # The application (auth-service app account) this caller is scoped to —
        # the isolation boundary, taken ONLY from the gateway's verified header
        # and never from a request body/query field. See
        # app/dependencies.py::resolve_account_id.
        request.state.account_id = account_id
        request.state.auth_method = "gateway"
        return await self._proceed(request, call_next, bearer)


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"code": "unauthorized", "message": detail}},
        headers={"WWW-Authenticate": "Bearer"},
    )

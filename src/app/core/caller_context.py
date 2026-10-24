"""The calling user's bearer token for the current request, and the middleware that binds it.

Every LLM and embedding call this service makes goes out through the API gateway to the
LLM gateway (app/rag/generation/llm_gateway.py), which authenticates it like any other
traffic. So the outbound call carries the *caller's* token, not a service credential:
that is what makes the gateway the one place where per-user rate limits, budgets and
metering are enforced. Same contract as makemerich-backend's app/auth/caller_context.py.

The token arrives on the inbound request: the API gateway forwards the caller's
``Authorization`` header to every upstream unchanged, alongside the identity headers this
service reads (app/core/auth.py). It is needed deep inside the RAG strategies, which take
no request, so a ``ContextVar`` carries it. Tasks started while serving a request (e.g.
background ingestion) copy the context at creation and keep the token.

This var holds a live credential, so two rules:
  * Always reset it. ``CallerTokenMiddleware`` binds it for exactly one request; anything
    else uses ``caller_token_scope``.
  * Never log it. ``has_caller_token`` lets callers branch on presence without the value.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_caller_token_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "knowledge_caller_token", default=""
)


def get_caller_token() -> str:
    """The bearer token of whoever is being served right now, or "" (scripts, tests)."""
    return _caller_token_var.get()


def has_caller_token() -> bool:
    return bool(_caller_token_var.get())


@contextmanager
def caller_token_scope(token: str) -> Iterator[None]:
    """Bind a caller's token for the duration of the ``with`` block."""
    var_token = _caller_token_var.set(token or "")
    try:
        yield
    finally:
        _caller_token_var.reset(var_token)


def bearer_from(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return ""
    return auth.split(" ", 1)[1].strip()


class CallerTokenMiddleware(BaseHTTPMiddleware):
    """Bind the inbound bearer token for the life of the request.

    Not validated here (that is the API gateway's job, done before the request arrives),
    only carried, so the LLM calls made while serving this request go out as this user.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        with caller_token_scope(bearer_from(request)):
            return await call_next(request)

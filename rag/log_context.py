"""Ambient correlation ID for the current logical request.

A single ``contextvars.ContextVar`` holding "the correlation ID for whatever
is running right now," set once at the top of the request (by HTTP
middleware, or by any in-process caller that wants one) and read by both the
structured log formatter (``app/logging_config.py``) and anything else that
wants to correlate its output with the request that triggered it. Everything
downstream — including work spawned by ``asyncio.create_task`` from inside the
request, since contextvars propagate through asyncio task creation — picks it
up with no parameter threading required.

Deliberately framework-free (contextvars is stdlib) and living in ``rag/``
rather than ``app/``: correlation-ID propagation is a portable core concern,
not an HTTP-only one. Ported from the LLM Gateway's ``features/log_context.py``
— same mechanism, same rationale.

Usage::

    from rag.log_context import bind_request_id, get_request_id

    token = bind_request_id("a1b2c3d4")
    try:
        ...  # everything in here, and everything it awaits, sees this ID
    finally:
        reset_request_id(token)
"""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "knowledge_service_request_id", default=""
)


def get_request_id() -> str:
    """The correlation ID for whatever is currently running, or "" if none
    was ever bound (e.g. a script importing rag/ directly with no
    correlation concept of its own)."""
    return _request_id_var.get()


def bind_request_id(request_id: str) -> contextvars.Token:
    """Bind a correlation ID for the current context. Returns a token for
    ``reset_request_id`` — prefer the ``request_id_scope`` context manager
    unless you specifically need the raw token."""
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


@contextmanager
def request_id_scope(request_id: str | None = None) -> Iterator[str]:
    """Bind a correlation ID for the duration of the ``with`` block.

    Mints a fresh one (same 16-hex-char shape as the HTTP middleware) when
    none is given, so in-process callers get correlated logs without needing
    to know the ID format.
    """
    resolved = request_id or uuid4().hex[:16]
    token = _request_id_var.set(resolved)
    try:
        yield resolved
    finally:
        _request_id_var.reset(token)

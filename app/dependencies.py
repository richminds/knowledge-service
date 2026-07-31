"""FastAPI dependencies shared across controllers.

Unlike the LLM Gateway, this service has no single "client" singleton to
build in the lifespan and hand out via ``Depends()`` — the RAG pipeline
(``rag/graph.py``) and the LLM Gateway HTTP clients (``rag/llm_gateway_client.py``)
are already module-level singletons/pure functions. What every route does
need is the authenticated identity set by ``AuthMiddleware``.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request


def get_principal(request: Request) -> str:
    """Authenticated identity for this request.

    Set by ``AuthMiddleware``: the API-key name, the JWT ``sub``, or
    "anonymous" when auth is disabled.
    """
    return getattr(request.state, "principal", "anonymous")


def get_role(request: Request) -> str:
    """Authorization role for this request.

    Set by ``AuthMiddleware``: an API key's configured role (``name:key:role``,
    default "user"), a JWT's ``role`` claim (default "user" when absent), or
    "admin" when auth is disabled entirely (open mode enforces nothing).
    """
    return getattr(request.state, "role", "admin")


def require_admin(role: str = Depends(get_role)) -> str:
    """Route dependency: 403 unless the caller's role is "admin".

    Use on routes that expose operational data (stats, config) rather than
    serving RAG traffic itself.
    """
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires the admin role.",
        )
    return role

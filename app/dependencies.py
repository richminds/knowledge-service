"""FastAPI dependencies shared across controllers.

Unlike the LLM Gateway, this service has no single "client" singleton to
build in the lifespan and hand out via ``Depends()`` — the RAG pipelines
(``rag/ingestion.py``, ``rag/retrieval.py``) and the LLM Gateway HTTP clients
(``rag/llm_gateway_client.py``) are already module-level singletons/pure
functions. What every route does need is the authenticated identity set by
``AuthMiddleware``.
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


def resolve_user_id(requested: str | None, request: Request) -> str:
    """Resolve the end-user identity for this request.

    For a JWT-authenticated end user, identity is exactly the verified
    token's ``sub`` — a request body/form cannot claim to be a different
    user; a mismatching ``user_id`` is rejected rather than silently
    overridden (a match, e.g. a client echoing back what it already knows, is
    a no-op). For a trusted service-to-service caller (static API key, or
    auth disabled entirely) the caller may assert an end-user ID explicitly —
    a calling application acting on behalf of many end-users (e.g. Portless)
    passes its own end-user's ID this way; a caller with no such concept just
    gets its own principal recorded. Mirrors ``resolve_org_id`` below.
    """
    principal = getattr(request.state, "principal", "anonymous")
    requested = (requested or "").strip()
    if getattr(request.state, "auth_method", None) == "jwt":
        if requested and requested != principal:
            raise HTTPException(
                status_code=403,
                detail="user_id does not match the authenticated token's subject.",
            )
        return principal
    return requested or principal


def resolve_org_id(requested: str | None, request: Request) -> str:
    """Resolve the tenant (org) for this request — the hard isolation boundary
    enforced in rag/authorization.py.

    For a JWT-authenticated end user, org_id comes ONLY from the verified
    token's ``org_id`` claim (set by AuthMiddleware) — never from a client-
    supplied field, since that field is exactly what separates one tenant's
    documents from another's. A request body/query ``org_id`` that doesn't
    match the token's is rejected rather than silently overridden. For a
    trusted service-to-service caller (static API key, or auth disabled
    entirely) the caller may assert org_id explicitly, same as
    ``resolve_user_id``.
    """
    requested = (requested or "").strip()
    if getattr(request.state, "auth_method", None) == "jwt":
        token_org_id = getattr(request.state, "org_id", "") or ""
        if requested and requested != token_org_id:
            raise HTTPException(
                status_code=403,
                detail="org_id does not match the authenticated token.",
            )
        return token_org_id
    return requested


def resolve_account_id(requested: str | None, request: Request) -> str:
    """Resolve the application (app account) for this request — the second
    isolation boundary, enforced in rag/authorization.py alongside org_id.

    A user can belong to several applications and picks one at sign-in
    (auth-service POST /auth/me/account), which bakes the choice into the
    token. So, exactly like ``resolve_org_id``, a JWT caller's account comes
    ONLY from the verified ``account_id`` claim and a mismatching client-
    supplied value is rejected; a trusted service-to-service caller (static
    API key, or auth disabled) may assert one explicitly.
    """
    requested = (requested or "").strip()
    if getattr(request.state, "auth_method", None) == "jwt":
        token_account_id = getattr(request.state, "account_id", "") or ""
        if requested and requested != token_account_id:
            raise HTTPException(
                status_code=403,
                detail="account_id does not match the authenticated token.",
            )
        return token_account_id
    return requested

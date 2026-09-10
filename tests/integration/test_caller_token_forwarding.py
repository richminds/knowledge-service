"""The caller's token reaching the LLM Gateway call, and never leaking past it.

This service routes every LLM and embedding call through the API Gateway so
that authentication, rate limits, budgets and metering apply to RAG traffic the
same way they apply to everything else. That only works if the call carries the
token of the user who caused it — otherwise the gateway sees one anonymous
service and every one of those controls is defeated.

Three properties are worth pinning, and they are the three ways this can go
wrong: the token must be attached when there is one, must NOT be silently
replaced by a service credential when there isn't, and must not survive its own
request.
"""
from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.auth import AuthMiddleware
from rag.caller_context import (
    caller_token_scope,
    get_caller_token,
    has_caller_token,
)
from rag.llm_gateway_client import _CallerToken, get_llm_client, reset_clients


# ---------------------------------------------------------------------------
# The ambient credential itself
# ---------------------------------------------------------------------------

def test_a_bound_token_is_visible_to_anything_the_request_awaits():
    with caller_token_scope("tok-abc"):
        assert get_caller_token() == "tok-abc"
        assert has_caller_token() is True


def test_the_token_does_not_outlive_its_scope():
    """A token left bound would be attached to whatever ran next on this
    context — someone else's request, with this user's credential."""
    with caller_token_scope("tok-abc"):
        pass
    assert get_caller_token() == ""
    assert has_caller_token() is False


def test_the_token_is_cleared_even_when_the_request_raises():
    with pytest.raises(ValueError):
        with caller_token_scope("tok-abc"):
            raise ValueError("pipeline blew up")
    assert get_caller_token() == ""


def test_scopes_nest_without_bleeding():
    with caller_token_scope("outer"):
        with caller_token_scope("inner"):
            assert get_caller_token() == "inner"
        assert get_caller_token() == "outer"


# ---------------------------------------------------------------------------
# What actually goes on the wire
# ---------------------------------------------------------------------------

def _echo_transport() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler), seen


async def test_the_callers_token_is_attached_to_the_outbound_call():
    transport, seen = _echo_transport()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw.test", auth=_CallerToken()
    ) as client:
        with caller_token_scope("tok-user-1"):
            await client.post("/v1/llm", json={})

    assert seen[0].headers["authorization"] == "Bearer tok-user-1"


async def test_no_token_means_no_authorization_header():
    """Deliberately not falling back to a service credential: substituting one
    would turn a user's call into an unattributed one, which is the exact
    behaviour routing through the gateway exists to stop. A visible 401 beats a
    quiet mis-attribution."""
    transport, seen = _echo_transport()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw.test", auth=_CallerToken()
    ) as client:
        await client.post("/v1/llm", json={})

    assert "authorization" not in seen[0].headers


async def test_one_pooled_client_carries_a_different_caller_each_call():
    """The point of resolving the credential per request rather than at
    construction: the connection pool is shared, the identity is not."""
    transport, seen = _echo_transport()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw.test", auth=_CallerToken()
    ) as client:
        for token in ("tok-alice", "tok-bob"):
            with caller_token_scope(token):
                await client.post("/v1/llm", json={})

    assert [r.headers["authorization"] for r in seen] == [
        "Bearer tok-alice",
        "Bearer tok-bob",
    ]


def test_the_real_client_is_wired_with_the_contextual_credential():
    """Guards against the singletons being rebuilt without the auth hook, which
    would silently send every call out unauthenticated."""
    reset_clients()
    try:
        client = get_llm_client()
        assert isinstance(client._client.auth, _CallerToken)
    finally:
        reset_clients()


async def test_the_gateway_prefix_is_preserved_when_joining_the_path():
    """RAG_GATEWAY_BASE_URL carries the gateway's /api/llm prefix, and the SDK
    posts to absolute-looking paths like /v1/llm. httpx must APPEND, not
    replace — replacing would send /v1/llm to the gateway root, which resolves
    to no route and 404s. Cheap to assert, silent and total if it ever changes.
    """
    transport, seen = _echo_transport()
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://gw.test/api/llm",
        auth=_CallerToken(),
    ) as client:
        await client.post("/v1/llm", json={})
        await client.post("/v1/embeddings", json={})

    assert [str(r.url) for r in seen] == [
        "https://gw.test/api/llm/v1/llm",
        "https://gw.test/api/llm/v1/embeddings",
    ]


def test_the_default_base_url_points_at_the_gateway_not_llm_gateway():
    """A default aimed straight at llm-gateway would quietly restore the bypass
    this routing exists to close.

    The declared default, not an instantiated RAGSettings: conftest sets
    RAG_GATEWAY_BASE_URL for the whole suite, and a real environment variable
    outranks the default in pydantic-settings — so instantiating would only
    ever assert what conftest chose.
    """
    from rag.config import RAGSettings

    declared = RAGSettings.model_fields["gateway_base_url"].default
    assert declared.rstrip("/").endswith("/api/llm")


# ---------------------------------------------------------------------------
# What the middleware binds, and what it deliberately does not
# ---------------------------------------------------------------------------

def _probe_app() -> TestClient:
    """The real AuthMiddleware in front of an endpoint that reports what it sees."""

    async def whoami(request):
        return PlainTextResponse(get_caller_token())

    app = Starlette(routes=[Route("/v1/probe", whoami)])
    app.add_middleware(AuthMiddleware)
    return TestClient(app)


def test_open_mode_still_forwards_the_token_the_gateway_sent():
    """Enforcement being off here does not mean the token is uninteresting: the
    gateway in front already verified it, and it is what identifies the user on
    this service's outbound calls."""
    response = _probe_app().get(
        "/v1/probe", headers={"Authorization": "Bearer tok-from-gateway"}
    )
    assert response.text == "tok-from-gateway"


def test_a_request_with_no_token_binds_nothing():
    assert _probe_app().get("/v1/probe").text == ""


def test_a_non_bearer_authorization_binds_nothing():
    response = _probe_app().get("/v1/probe", headers={"Authorization": "Basic abc"})
    assert response.text == ""


def test_an_api_key_header_binds_nothing():
    """Static service API keys were removed entirely (see
    app/middleware/auth.py). X-API-Key is now an ordinary unrecognised header
    — it authenticates nothing here and contributes nothing to forward."""
    response = _probe_app().get("/v1/probe", headers={"X-API-Key": "sk-secret"})
    assert response.text == ""

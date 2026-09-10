"""Singleton HTTP clients for the LLM Gateway, reached through the API Gateway.

One ``RemoteLLMClient`` and one ``GatewayEmbeddingsClient`` are built lazily
and reused for the lifetime of the process — each owns its own connection
pool, so building one per call would open a fresh pool per request, exactly
the mistake the LLM Gateway's own ``app/main.py`` lifespan avoids for its
in-process ``LLMClient``.

Built from ``RAG_GATEWAY_*`` (``rag/config.py``): this service holds no
provider credential of its own and never imports a provider SDK — every
embedding and chat call in this codebase (``rag/llm.py``, ``rag/embeddings.py``,
``rag/security.py``, ``rag/evaluation.py``) crosses the network through the two
clients built here.

**Through the API Gateway, not straight at llm-gateway.**
``RAG_GATEWAY_BASE_URL`` points at the gateway's ``/api/llm`` prefix, which
strips it and forwards to llm-gateway. The extra hop buys one thing, and it is
the reason for it: the gateway is then the single place where authentication,
per-user and per-account rate limits, budgets and usage metering are enforced.
Calling llm-gateway directly — as this module used to — meant RAG traffic
arrived as one static service key, so every user's LLM spend landed in the same
bucket, no per-user budget could apply to it, and the gateway's metering never
saw it at all.

**Which credential goes out.** The gateway authenticates this call like any
other, so it must carry a token it can introspect: the *caller's* own, taken
from ``rag/caller_context.py``. ``_CallerToken`` below is consulted per request
rather than baked in at construction, which is what lets one pooled client
serve a different user on every call.

Two consequences worth knowing before changing anything here:

  * **A request with no caller token gets none attached.** The CLI, the tests
    and any in-process caller have no HTTP request behind them. Those calls
    will 401, which is correct — there is no user to bill or budget. Mint a
    service token minted by auth-service and bind it with
    ``rag.caller_context.caller_token_scope`` for that kind of work.
  * **A background job carries a snapshot.** Ingestion re-binds the token that
    started it, so a job outliving that token's expiry will start failing its
    embedding calls. See ``app/services/ingest_service.py``.
"""
from __future__ import annotations

import logging
from collections.abc import Generator

import httpx

from .caller_context import get_caller_token
from .config import settings
from .llm_gateway_sdk import GatewayEmbeddingsClient, RemoteLLMClient

logger = logging.getLogger(__name__)

_llm_client: RemoteLLMClient | None = None
_embeddings_client: GatewayEmbeddingsClient | None = None


class _CallerToken(httpx.Auth):
    """Attaches the current caller's bearer token to each outbound request.

    An ``httpx.Auth`` is consulted per request, which is the whole point: the
    clients here are process-wide singletons holding connection pools, so a
    credential fixed at construction could only ever be a service identity.
    This reads the ambient token instead, so the pool is shared while the
    identity on the wire is the user actually being served.

    Sending nothing when nothing is bound is deliberate. Substituting a service
    credential would silently turn a user's call into an unattributed one —
    exactly the behaviour this module was changed to stop — so the request goes
    out unauthenticated and the gateway rejects it, which is a visible failure
    rather than a quiet mis-attribution.
    """

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        token = get_caller_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


def get_llm_client() -> RemoteLLMClient:
    """Return the process-wide RemoteLLMClient, building it on first use."""
    global _llm_client
    if _llm_client is None:
        _llm_client = RemoteLLMClient(
            base_url=settings.gateway_base_url,
            timeout=settings.gateway_timeout_seconds,
            auth=_CallerToken(),
        )
    return _llm_client


def get_embeddings_client() -> GatewayEmbeddingsClient:
    """Return the process-wide GatewayEmbeddingsClient, building it on first use."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GatewayEmbeddingsClient(
            base_url=settings.gateway_base_url,
            timeout=settings.gateway_timeout_seconds,
            auth=_CallerToken(),
        )
    return _embeddings_client


async def aclose_llm_client() -> None:
    """Close the pooled RemoteLLMClient's connections (call from app shutdown)."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.aclose()
        _llm_client = None


def reset_clients() -> None:
    """Drop both singletons so the next call rebuilds them from current settings.

    For tests, which change ``RAG_GATEWAY_BASE_URL`` between cases; a client
    built at first use would otherwise pin the first test's configuration for
    the whole session.
    """
    global _llm_client, _embeddings_client
    _llm_client = None
    _embeddings_client = None

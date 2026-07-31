"""Singleton HTTP clients for the configured LLM Gateway instance.

One ``RemoteLLMClient`` and one ``GatewayEmbeddingsClient`` are built lazily
and reused for the lifetime of the process — each owns its own connection
pool, so building one per call would open a fresh pool per request, exactly
the mistake the LLM Gateway's own ``app/main.py`` lifespan avoids for its
in-process ``LLMClient``.

Built from ``RAG_GATEWAY_*`` (``rag/config.py``): this service holds no
provider credential of its own and never imports a provider SDK — every
embedding and chat call in this codebase (``rag/llm.py``, ``rag/embeddings.py``,
``rag/security.py``, ``rag/evaluation.py``) crosses the network to the gateway
through the two clients built here.
"""
from __future__ import annotations

from .config import settings
from .llm_gateway_sdk import GatewayEmbeddingsClient, RemoteLLMClient

_llm_client: RemoteLLMClient | None = None
_embeddings_client: GatewayEmbeddingsClient | None = None


def get_llm_client() -> RemoteLLMClient:
    """Return the process-wide RemoteLLMClient, building it on first use."""
    global _llm_client
    if _llm_client is None:
        _llm_client = RemoteLLMClient(
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout=settings.gateway_timeout_seconds,
        )
    return _llm_client


def get_embeddings_client() -> GatewayEmbeddingsClient:
    """Return the process-wide GatewayEmbeddingsClient, building it on first use."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GatewayEmbeddingsClient(
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout=settings.gateway_timeout_seconds,
        )
    return _embeddings_client


async def aclose_llm_client() -> None:
    """Close the pooled RemoteLLMClient's connections (call from app shutdown)."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.aclose()
        _llm_client = None

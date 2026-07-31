"""Embeddings for the RAG pipeline — routed through the LLM Gateway over HTTP.

Every embedding call goes through the vendored LLM Gateway SDK
(``rag.llm_gateway_sdk.GatewayEmbeddingsClient``), which owns nothing but an
HTTP connection to a deployed LLM Gateway instance (``RAG_GATEWAY_BASE_URL``).
This service holds no provider API key and imports no provider SDK — the
gateway owns the model/provider/credential config *and* the actual provider
call / offline fallback. See ``rag/config.py`` for ``RAG_GATEWAY_*`` and the
LLM Gateway's own ``LLM_EMBEDDING_*`` settings for the single source of truth
on which model is actually served.
"""
from __future__ import annotations

from .llm_gateway_client import get_embeddings_client
from .llm_gateway_sdk import GatewayEmbeddingsClient

__all__ = ["get_embeddings", "GatewayEmbeddingsClient"]


def get_embeddings() -> GatewayEmbeddingsClient:
    """Return the service's embedding client.

    Delegates entirely to the LLM Gateway over HTTP — this package passes no
    model config and does not import any provider SDK.
    """
    return get_embeddings_client()

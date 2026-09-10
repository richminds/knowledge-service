"""Operational visibility — chunk counts and resolved (non-secret) configuration."""
from __future__ import annotations

from rag.config import settings as rag_settings
from rag.embeddings import get_embeddings
from rag.file_store import get_file_store
from rag.vector_store import VectorStore

from ..config import gateway_settings
from ..models.stats_model import ConfigResponse, StatsResponse


async def get_stats() -> StatsResponse:
    """Return indexed-chunk count plus the active file-store backend and
    graph/web-search retrieval status — reflects ground truth (whichever .env
    actually won) rather than requiring anyone to go spelunking through config
    files."""
    embeddings = get_embeddings()
    vector_store = VectorStore(embeddings)
    total = await vector_store.count()
    return StatsResponse(
        total_chunks=total,
        collection=rag_settings.chunks_collection,
        db=rag_settings.mongo_db_name,
        file_store_backend=get_file_store().backend,
        graph_retrieval_enabled=rag_settings.graph_retrieval_enabled,
        graph_db_backend=rag_settings.graph_db_backend,
        web_search_enabled=rag_settings.web_search_enabled,
        gateway_base_url=rag_settings.gateway_base_url,
    )


def get_config() -> ConfigResponse:
    """Resolved, non-secret configuration — mirrors the LLM Gateway's own
    GET /v1/config. Never includes jwt_secret."""
    return ConfigResponse(
        service={
            "environment": gateway_settings.environment,
            "log_format": gateway_settings.log_format,
            "docs_enabled": gateway_settings.docs_enabled,
            "cors_origins": gateway_settings.parsed_cors_origins(),
            "auth_configured": rag_settings.auth_enabled,
        },
        rag={
            "mongo_db_name": rag_settings.mongo_db_name,
            "chunks_collection": rag_settings.chunks_collection,
            "vector_index_name": rag_settings.vector_index_name,
            "embedding_dimensions": rag_settings.embedding_dimensions,
            "chunk_size": rag_settings.chunk_size,
            "chunk_overlap": rag_settings.chunk_overlap,
            "retrieval_top_k": rag_settings.retrieval_top_k,
            "use_cross_encoder_reranker": rag_settings.use_cross_encoder_reranker,
            "advanced_eval_provider": rag_settings.advanced_eval_provider,
            "graph_retrieval_enabled": rag_settings.graph_retrieval_enabled,
            "web_search_enabled": rag_settings.web_search_enabled,
            "gateway_base_url": rag_settings.gateway_base_url,
        },
    )

from app.core.config import Settings
from app.rag.vectorstore.base import VectorStore

_vector_store_singleton: VectorStore | None = None


def get_vector_store(settings: Settings) -> VectorStore:
    """The process-wide VectorStore: one provider, or a MultiVectorStore when several are
    configured (VECTOR_STORE_PROVIDERS) - writes go to all, reads to the selected one."""
    global _vector_store_singleton
    if _vector_store_singleton is not None:
        return _vector_store_singleton

    providers = settings.active_vector_providers
    if len(providers) == 1:
        _vector_store_singleton = _build_vector_store(settings, providers[0])
    else:
        from app.rag.vectorstore.multi import MultiVectorStore

        _vector_store_singleton = MultiVectorStore(
            {name: _build_vector_store(settings, name) for name in providers},
            default=settings.vector_store_provider,
        )
    return _vector_store_singleton


def vector_stores(settings: Settings) -> dict[str, VectorStore]:
    """Every underlying store by provider name (startup index creation, diagnostics)."""
    from app.rag.vectorstore.multi import MultiVectorStore

    store = get_vector_store(settings)
    if isinstance(store, MultiVectorStore):
        return store.stores
    return {settings.vector_store_provider: store}


def _semantic_reranker(settings: Settings):
    """Atlas has no built-in semantic ranker, so semantic_hybrid reranks in the app
    (MONGODB_SEMANTIC_RERANKER): llm = the utility model, cohere = a cross-encoder endpoint."""
    choice = settings.mongodb_semantic_reranker
    if settings.retrieval_mode != "semantic_hybrid" or choice == "none":
        return None
    from app.rag.retrieval.reranker import CohereReranker, LlmReranker

    if choice == "cohere":
        return CohereReranker(
            settings.reranker_endpoint,
            settings.reranker_api_key.get_secret_value(),
            settings.reranker_model,
        )
    from app.api.deps import get_chat_client

    return LlmReranker(get_chat_client(settings))


def _build_vector_store(settings: Settings, provider: str) -> VectorStore:
    if provider == "mongo":
        from app.rag.vectorstore.mongo_store import MongoVectorStore

        return MongoVectorStore(
            url=settings.mongodb_url.get_secret_value(),
            database=settings.mongodb_database,
            collection=settings.mongodb_chunks_collection,
            vector_index=settings.mongodb_vector_index,
            search_index=settings.mongodb_search_index,
            embedding_dim=settings.embedding_dimensions,
            retrieval_mode=settings.retrieval_mode,
            text_analyzer=settings.mongodb_text_analyzer,
            num_candidates_factor=settings.mongodb_num_candidates_factor,
            semantic_reranker=_semantic_reranker(settings),
            semantic_candidates=settings.mongodb_semantic_candidates,
        )

    raise ValueError(f"Unsupported vector store provider: {provider}")

from app.rag.ingestion.embeddings import EmbeddingClient
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult, VectorStore


class VectorRetriever:
    """Embeds a query and searches the index (vector or hybrid, per vector store config)."""

    def __init__(self, embedding_client: EmbeddingClient, vector_store: VectorStore) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    async def embed(self, text: str) -> list[float]:
        """Query embedding with the same model/version as ingestion (S-18)."""
        return await self._embedding_client.embed_text(text)

    async def retrieve(
        self, query: str, filters: RetrievalFilters, top_k: int
    ) -> list[VectorSearchResult]:
        query_embedding = await self._embedding_client.embed_text(query)
        return await self._vector_store.search(query_embedding, filters, top_k, query_text=query)

    async def retrieve_by_vector(
        self, embedding: list[float], filters: RetrievalFilters, top_k: int
    ) -> list[VectorSearchResult]:
        """Dense-only search with a precomputed embedding (HyDE, S-24)."""
        return await self._vector_store.search(embedding, filters, top_k)

    async def keyword_search(
        self, query: str, filters: RetrievalFilters, top_k: int
    ) -> list[VectorSearchResult]:
        """BM25-only search: exact ids, codes and names (S-21, keyword_search tool A-08)."""
        return await self._vector_store.search(None, filters, top_k, query_text=query)

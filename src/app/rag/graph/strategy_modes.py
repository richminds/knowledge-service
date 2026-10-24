"""Run a specific GraphRagStrategy mode (used by the agent's graph tools, A-09..A-11)."""

from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult


async def run_graph_mode(
    strategy, mode: str, query: str, filters: RetrievalFilters, top_k: int
) -> list[VectorSearchResult]:
    return (await strategy.retrieve(query, filters, top_k, mode=mode)).contexts

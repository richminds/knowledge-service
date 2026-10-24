"""Hybrid + graph RAG: hybrid search (dense vector + BM25) and knowledge-graph retrieval run
side by side, and their contexts are fused with Reciprocal Rank Fusion (C-168).

  hybrid search       chunks that match the question's wording and meaning
  graph retrieval     entity facts, relations, community reports and the chunks behind them
  RRF fusion          chunks from both sides are fused, so a chunk found by both rises;
                      the graph's own passages (facts, community reports) are kept in front

The graph side is best-effort: when it fails or finds nothing, the hybrid results are used
on their own (the trace says so). Reranking and token budgeting happen afterwards in the
shared ContextAssembler.
"""

import asyncio

from app.core.config import RagMode
from app.core.exceptions import GenerationError, GraphStoreError
from app.core.logging import get_logger
from app.rag.context.fusion import rrf_fuse
from app.rag.strategies.base import History, RagStrategy, RetrievalOutcome, traced
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult

logger = get_logger(__name__)


class HybridGraphRagStrategy(RagStrategy):
    mode = RagMode.HYBRID_GRAPH

    def __init__(self, hybrid: RagStrategy, graph: RagStrategy) -> None:
        """`graph` should have no semantic fallback: the hybrid side already covers it."""
        self._hybrid = hybrid
        self._graph = graph

    async def _graph_contexts(
        self, query: str, filters: RetrievalFilters, top_k: int, history: History
    ) -> RetrievalOutcome | None:
        try:
            return await self._graph.retrieve(query, filters, top_k, history)
        except (GraphStoreError, GenerationError) as exc:
            logger.warning("hybrid_graph_graph_side_failed", error=exc.message)
            return None

    async def retrieve(
        self, query: str, filters: RetrievalFilters, top_k: int, history: History = ()
    ) -> RetrievalOutcome:
        hybrid, graph = await asyncio.gather(
            self._hybrid.retrieve(query, filters, top_k, history),
            self._graph_contexts(query, filters, top_k, history),
        )
        outcome = RetrievalOutcome(contexts=[])
        outcome.trace.extend(hybrid.trace)
        graph_contexts = graph.contexts if graph is not None else []
        if graph is not None:
            outcome.trace.extend(graph.trace)

        # Graph facts, community reports and query results are extra evidence, not chunks:
        # they are kept on top of the fused chunks so top_k chunks still reach the assembler.
        graph_evidence = [c for c in graph_contexts if is_graph_evidence(c)]
        graph_chunks = [c for c in graph_contexts if not is_graph_evidence(c)]
        with traced(outcome.trace, "fuse_hybrid_graph") as info:
            lists = [c for c in (hybrid.contexts, graph_chunks) if c]
            fused = rrf_fuse(lists) if len(lists) > 1 else (lists[0] if lists else [])
            info.update(
                hybrid_chunks=len(hybrid.contexts),
                graph_chunks=len(graph_chunks),
                graph_evidence=len(graph_evidence),
                graph_failed=graph is None,
            )
        outcome.contexts = graph_evidence + fused[:top_k]
        return outcome


def is_graph_evidence(context: VectorSearchResult) -> bool:
    """A passage the graph side wrote itself (facts, community report, query result)."""
    return context.source == "knowledge-graph" or context.metadata.get("retriever") in (
        "graph-global",
        "graph-cypher",
    )

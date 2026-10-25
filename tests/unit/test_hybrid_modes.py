"""semantic (vector only), hybrid (vector + BM25) and hybrid_graph (hybrid + knowledge graph)."""

import pytest

from app.core.config import RagMode, Settings
from app.core.exceptions import GraphStoreError
from app.rag.ingestion.pipeline import LoadedDocument
from app.rag.strategies.base import RagStrategy, RetrievalOutcome
from app.rag.strategies.hybrid_graph import HybridGraphRagStrategy, is_graph_evidence
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult

DEFAULT = RetrievalFilters(collection="default")
ORG_DOC = LoadedDocument(
    source="org.md",
    text="Alice manages the Finance department. Finance approves Travel requests.",
)
LEAVE_DOC = LoadedDocument(
    source="leave.md", text="Annual leave policy: staff receive 25 days of leave per year."
)


async def _ingest(ingestion_pipeline):
    await ingestion_pipeline.run_for_documents(
        [ORG_DOC, LEAVE_DOC],
        "default",
        targets=["vector", "graph"],
        metadata={"account_id": "acct-test"},
    )


def _hit(chunk_id: str, source: str = "doc.md", retriever: str = "vector") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        content=chunk_id,
        score=1.0,
        source=source,
        metadata={"retriever": retriever},
    )


class _Fixed(RagStrategy):
    mode = RagMode.GRAPH

    def __init__(self, contexts=(), error: Exception | None = None) -> None:
        self._contexts = list(contexts)
        self._error = error

    async def retrieve(self, query, filters, top_k, history=()):
        if self._error is not None:
            raise self._error
        return RetrievalOutcome(contexts=list(self._contexts))


# -- semantic vs hybrid ------------------------------------------------------------------
async def test_semantic_searches_vectors_only(ingestion_pipeline, strategies, vector_store):
    await _ingest(ingestion_pipeline)
    vector_store.search_calls.clear()

    outcome = await strategies[RagMode.SEMANTIC].retrieve("annual leave days", DEFAULT, 3)

    assert outcome.contexts
    assert all(c["query_text"] is None and c["hybrid"] is False for c in vector_store.search_calls)
    assert outcome.trace[0].detail["search"] == "vector"


async def test_hybrid_adds_keyword_search(ingestion_pipeline, strategies, vector_store):
    await _ingest(ingestion_pipeline)
    vector_store.search_calls.clear()

    outcome = await strategies[RagMode.HYBRID].retrieve("annual leave days", DEFAULT, 3)

    assert outcome.contexts[0].source == "leave.md"
    assert vector_store.search_calls
    assert all(c["query_text"] and c["hybrid"] is True for c in vector_store.search_calls)
    assert outcome.trace[0].detail["search"] == "hybrid"


def test_semantic_strategy_rejects_other_modes(strategies):
    from app.rag.strategies.semantic import SemanticRagStrategy

    with pytest.raises(ValueError, match="semantic or hybrid"):
        SemanticRagStrategy(None, None, mode=RagMode.GRAPH)


# -- hybrid + graph ----------------------------------------------------------------------
async def test_hybrid_graph_returns_graph_facts_and_chunks(ingestion_pipeline, strategies):
    await _ingest(ingestion_pipeline)

    outcome = await strategies[RagMode.HYBRID_GRAPH].retrieve("Who runs Finance?", DEFAULT, 3)

    assert outcome.contexts[0].source == "knowledge-graph"
    assert any(c.source == "org.md" for c in outcome.contexts[1:])
    fuse = next(s for s in outcome.trace if s.step == "fuse_hybrid_graph")
    assert fuse.detail["graph_evidence"] >= 1 and fuse.detail["graph_failed"] is False


async def test_hybrid_graph_fuses_chunks_from_both_sides():
    hybrid = _Fixed([_hit("a"), _hit("b"), _hit("c")])
    graph = _Fixed([_hit("graph:x", source="knowledge-graph", retriever="graph"), _hit("c")])

    outcome = await HybridGraphRagStrategy(hybrid, graph).retrieve("q", DEFAULT, 2)

    # facts first; "c" (found by both) outranks "a" (hybrid only); top_k caps the chunks
    assert [c.chunk_id for c in outcome.contexts] == ["graph:x", "c", "a"]


async def test_hybrid_graph_falls_back_to_hybrid_when_graph_fails():
    hybrid = _Fixed([_hit("a"), _hit("b")])
    graph = _Fixed(error=GraphStoreError("graph DB is down"))

    outcome = await HybridGraphRagStrategy(hybrid, graph).retrieve("q", DEFAULT, 5)

    assert [c.chunk_id for c in outcome.contexts] == ["a", "b"]
    fuse = next(s for s in outcome.trace if s.step == "fuse_hybrid_graph")
    assert fuse.detail["graph_failed"] is True


def test_graph_evidence_is_told_apart_from_graph_linked_chunks():
    assert is_graph_evidence(_hit("graph:x", source="knowledge-graph", retriever="graph"))
    assert is_graph_evidence(_hit("report-1", source="r.md", retriever="graph-global"))
    assert not is_graph_evidence(_hit("chunk-1", source="org.md", retriever="graph"))


# -- vector store override -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("configured", "hybrid", "expected"),
    [
        ("vector", None, False),
        ("hybrid", None, True),
        ("vector", True, True),
        ("hybrid", False, False),
        ("semantic_hybrid", True, True),
    ],
)
def test_per_call_hybrid_overrides_retrieval_mode(configured, hybrid, expected):
    from app.rag.vectorstore.mongo_store import MongoVectorStore

    store = MongoVectorStore.__new__(MongoVectorStore)
    store._retrieval_mode = configured
    assert store._keyword_with_vectors(hybrid) is expected


# -- configuration -------------------------------------------------------------------------
def test_hybrid_graph_needs_both_stores():
    settings = Settings(rag_enabled_modes=["hybrid_graph"], rag_mode="hybrid_graph")

    assert settings.vector_enabled and settings.graph_enabled


def test_hybrid_needs_no_graph_store():
    settings = Settings(rag_enabled_modes=["semantic", "hybrid"], rag_mode="hybrid")

    assert settings.vector_enabled and not settings.graph_enabled

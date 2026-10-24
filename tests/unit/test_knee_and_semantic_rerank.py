"""Knee-point cutoff (Kneedle) and the MongoDB store's app-side semantic reranking."""

import pytest

from app.rag.context.assembler import AssemblyOptions, ContextAssembler
from app.rag.context.fusion import knee_cutoff, knee_point
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult
from app.rag.vectorstore.mongo_store import MongoVectorStore


def _r(chunk_id: str, score: float, retriever: str = "vector") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id, content=chunk_id, score=score, source="a.md",
        metadata={"retriever": retriever},
    )


# -- knee_point ---------------------------------------------------------------------------------
def test_knee_after_a_sharp_drop_keeps_the_strong_group():
    assert knee_point([0.9, 0.85, 0.2, 0.18, 0.15]) == 2


def test_knee_before_a_cliff_keeps_everything_above_it():
    assert knee_point([0.9, 0.88, 0.86, 0.2]) == 3


def test_one_clear_winner():
    assert knee_point([3.6, 1.0, 0.9, 0.8]) == 1


@pytest.mark.parametrize(
    "scores",
    [
        [0.9, 0.8, 0.7, 0.6, 0.5],  # evenly spread: no knee
        [0.5, 0.5, 0.5],  # all equal
        [0.9, 0.1],  # too few points to have a knee
        [],
    ],
)
def test_no_knee(scores):
    assert knee_point(scores) is None


def test_sensitivity_controls_how_sharp_a_knee_must_be():
    scores = [0.9, 0.75, 0.55, 0.4, 0.35]
    assert knee_point(scores, sensitivity=0.5) is None
    assert knee_point(scores, sensitivity=0.05) is not None


def test_knee_point_sorts_its_input():
    assert knee_point([0.2, 0.9, 0.15, 0.85, 0.18]) == 2


# -- knee_cutoff --------------------------------------------------------------------------------
def test_cutoff_keeps_order_and_pins_graph_contexts():
    results = [
        _r("graph-facts", 1.0, "graph"),
        _r("c3", 0.18),
        _r("c1", 0.9),
        _r("c2", 0.85),
        _r("c4", 0.15),
        _r("c5", 0.2),
    ]
    kept = knee_cutoff(results)
    assert [r.chunk_id for r in kept] == ["graph-facts", "c1", "c2"]


def test_cutoff_leaves_results_alone_without_a_knee():
    results = [_r("a", 0.9), _r("b", 0.8), _r("c", 0.7)]
    assert knee_cutoff(results) == results


async def test_assembler_applies_the_knee_only_when_enabled():
    candidates = [_r("c1", 3.4), _r("c2", 3.1), _r("c3", 0.6), _r("c4", 0.5), _r("c5", 0.4)]
    trace: list = []
    off = await ContextAssembler(options=AssemblyOptions(top_n=5)).assemble("q", candidates, trace)
    on = await ContextAssembler(options=AssemblyOptions(top_n=5, knee=True)).assemble(
        "q", candidates, trace
    )
    assert len(off) == 5
    assert {r.chunk_id for r in on} == {"c1", "c2"}


# -- MongoDB semantic reranking ----------------------------------------------------------------
class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query, results, top_n):
        self.calls.append((query, [r.chunk_id for r in results], top_n))
        return [r.__class__(**{**r.__dict__, "score": 4.0 - i}) for i, r in
                enumerate(reversed(results))][:top_n]


def _store(mode: str, reranker) -> MongoVectorStore:
    store = MongoVectorStore(
        url="mongodb://test.invalid", database="d", retrieval_mode=mode,
        semantic_reranker=reranker, semantic_candidates=5,
    )

    async def vector(embedding, filters, kinds, top_k):
        return [_r(f"v{i}", 1.0 - i / 10) for i in range(top_k)]

    async def keyword(text, filters, kinds, top_k):
        return [_r(f"k{i}", 1.0 - i / 10) for i in range(top_k)]

    store._vector_search = vector  # type: ignore[method-assign]
    store._keyword_search = keyword  # type: ignore[method-assign]
    return store


async def test_semantic_hybrid_reranks_fused_candidates_on_mongo():
    reranker = RecordingReranker()
    out = await _store("semantic_hybrid", reranker).search(
        [0.1], RetrievalFilters(collection="c"), 3, query_text="annual leave"
    )
    [(query, candidates, top_n)] = reranker.calls
    assert query == "annual leave" and top_n == 3
    assert len(candidates) == 5  # MONGODB_SEMANTIC_CANDIDATES, more than top_k
    assert len(out) == 3 and out[0].score == 4.0  # the reranker's order and scores win


async def test_plain_hybrid_does_not_rerank():
    reranker = RecordingReranker()
    out = await _store("hybrid", reranker).search(
        [0.1], RetrievalFilters(collection="c"), 3, query_text="annual leave"
    )
    assert reranker.calls == [] and len(out) == 3


def test_semantic_reranker_is_only_built_for_semantic_hybrid(monkeypatch):
    from app.core.config import Settings
    from app.rag.retrieval.reranker import LlmReranker
    from app.rag.vectorstore.factory import _semantic_reranker

    assert _semantic_reranker(Settings(retrieval_mode="hybrid")) is None
    assert _semantic_reranker(
        Settings(retrieval_mode="semantic_hybrid", mongodb_semantic_reranker="none")
    ) is None
    from app.api import deps

    monkeypatch.setattr(deps, "get_chat_client", lambda settings: object())
    assert isinstance(
        _semantic_reranker(Settings(retrieval_mode="semantic_hybrid")), LlmReranker
    )

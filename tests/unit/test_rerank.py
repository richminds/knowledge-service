from langchain_core.documents import Document

from rag.rerank import local_rerank, select_by_knee_point
from rag.vector_store import ScoredDocument


def _sd(text: str, score: float, **meta) -> ScoredDocument:
    return ScoredDocument(
        document=Document(page_content=text, metadata=meta), score=score, source="semantic"
    )


def test_local_rerank_favours_lexical_overlap():
    candidates = [
        _sd("completely unrelated filler content", 0.9),
        _sd("apples and oranges are fruit", 0.5),
    ]
    ranked = local_rerank("apples oranges", candidates)
    assert ranked[0].document.page_content == "apples and oranges are fruit"


def test_local_rerank_empty_input():
    assert local_rerank("anything", []) == []


def test_select_by_knee_point_returns_short_lists_unchanged():
    candidates = [_sd("a", 1.0), _sd("b", 0.5)]
    assert select_by_knee_point(candidates, min_k=3) == candidates


def test_select_by_knee_point_respects_bounds():
    scores = [1.0, 0.95, 0.9, 0.1, 0.05, 0.01]
    candidates = [_sd(f"doc{i}", score) for i, score in enumerate(scores)]
    selected = select_by_knee_point(candidates, min_k=2, max_k=4)
    assert 2 <= len(selected) <= 4


def test_select_by_knee_point_uniform_scores_uses_max_k():
    candidates = [_sd(f"doc{i}", 0.5) for i in range(6)]
    selected = select_by_knee_point(candidates, min_k=2, max_k=4)
    assert len(selected) == 4

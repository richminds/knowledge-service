"""Candidate fusion and selection utilities (C-167..C-174, S-22, S-28).

All functions are pure (no I/O) and work on VectorSearchResult lists from any retriever.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import replace

import tiktoken

from app.rag.vectorstore.base import VectorSearchResult

_ENCODING = tiktoken.get_encoding("cl100k_base")
_TOKEN_RE = re.compile(r"\w+")


def rrf_fuse(result_lists: list[list[VectorSearchResult]], k: int = 60) -> list[VectorSearchResult]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)). Scale-free, so it merges results
    from retrievers whose scores are not comparable (vector, BM25, graph) (C-168)."""
    scores: dict[str, float] = {}
    first: dict[str, VectorSearchResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(result.chunk_id, result)
    fused = [
        replace(
            first[cid],
            score=score,
            metadata={**first[cid].metadata, "fused_from": str(len(result_lists))},
        )
        for cid, score in scores.items()
    ]
    return sorted(fused, key=lambda r: r.score, reverse=True)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedupe(
    results: list[VectorSearchResult], near_duplicate: float = 0.9
) -> list[VectorSearchResult]:
    """Drop repeated chunk ids and near-identical passages, keeping the best-ranked (C-169)."""
    kept: list[VectorSearchResult] = []
    kept_tokens: list[set[str]] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk_id in seen:
            continue
        tokens = _tokens(result.content)
        if any(jaccard(tokens, other) >= near_duplicate for other in kept_tokens):
            continue
        seen.add(result.chunk_id)
        kept.append(result)
        kept_tokens.append(tokens)
    return kept


def mmr(
    results: list[VectorSearchResult], top_n: int, lambda_: float = 0.7
) -> list[VectorSearchResult]:
    """Maximal Marginal Relevance with lexical similarity: trade relevance for diversity so
    near-duplicate passages don't crowd the context (S-28). Input must be relevance-ordered."""
    if len(results) <= 1:
        return results[:top_n]
    max_score = max(r.score for r in results) or 1.0
    tokens = [_tokens(r.content) for r in results]
    selected: list[int] = []
    remaining = list(range(len(results)))
    while remaining and len(selected) < top_n:

        def value(i: int) -> float:
            relevance = results[i].score / max_score
            redundancy = max((jaccard(tokens[i], tokens[j]) for j in selected), default=0.0)
            return lambda_ * relevance - (1 - lambda_) * redundancy

        best = max(remaining, key=value)
        selected.append(best)
        remaining.remove(best)
    return [results[i] for i in selected]


def score_cutoff(
    results: list[VectorSearchResult], min_score: float = 0.0, min_relative: float = 0.0
) -> list[VectorSearchResult]:
    """Keep results above an absolute floor and above a share of the best score (C-171, S-27).
    The best result is always kept so a weak-but-only answer still reaches the grader."""
    if not results:
        return results
    best = max(r.score for r in results)
    floor = max(min_score, best * min_relative)
    kept = [r for r in results if r.score >= floor]
    return kept or results[:1]


def lost_in_the_middle(results: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """Place the strongest passages at the start and end of the context (C-173):
    ranks 1,3,5,... first, then ...,6,4,2."""
    front = results[0::2]
    back = results[1::2]
    return front + back[::-1]


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def fit_token_budget(
    results: list[VectorSearchResult], max_tokens: int
) -> list[VectorSearchResult]:
    """Keep passages in order until the budget is used; truncate the one crossing it (C-174)."""
    kept: list[VectorSearchResult] = []
    used = 0
    for result in results:
        tokens = _ENCODING.encode(result.content)
        if used + len(tokens) <= max_tokens:
            kept.append(result)
            used += len(tokens)
            continue
        room = max_tokens - used
        if room > 50:  # a useful fragment
            kept.append(replace(result, content=_ENCODING.decode(tokens[:room]) + " ..."))
        break
    return kept


def knee_point(scores: Sequence[float], sensitivity: float = 0.2) -> int | None:
    """Kneedle: how many of the (descending) `scores` come before the knee, or None.

    Scores and ranks are normalised to [0, 1] and compared with the straight line from the
    best to the worst score. A curve that drops sharply then flattens (convex) has its knee at
    the point furthest below that line; one that stays high then falls off a cliff (concave)
    at the point furthest above it. `sensitivity` is the minimum distance (0-1) that counts as
    a knee, so evenly spread scores are left alone.
    """
    values = sorted(scores, reverse=True)
    n = len(values)
    if n < 3 or values[0] - values[-1] <= 0:
        return None
    low, span = values[-1], values[0] - values[-1]
    diffs = [(1 - i / (n - 1)) - (v - low) / span for i, v in enumerate(values)]
    convex = max(range(n), key=lambda i: diffs[i])
    concave = min(range(n), key=lambda i: diffs[i])
    if diffs[convex] >= -diffs[concave]:
        return convex if diffs[convex] >= sensitivity else None
    return concave + 1 if -diffs[concave] >= sensitivity else None


def knee_cutoff(
    results: list[VectorSearchResult],
    sensitivity: float = 0.2,
    min_keep: int = 1,
    pinned: Callable[[VectorSearchResult], bool] = lambda r: False,
) -> list[VectorSearchResult]:
    """Drop the results after the score knee, keeping their order. `pinned` results (graph
    fact summaries, whose scores are on another scale) are always kept."""
    ranked = sorted((r for r in results if not pinned(r)), key=lambda r: r.score, reverse=True)
    k = knee_point([r.score for r in ranked], sensitivity)
    if k is None:
        return results
    keep = {id(r) for r in ranked[: max(k, min_keep)]}
    return [r for r in results if pinned(r) or id(r) in keep]

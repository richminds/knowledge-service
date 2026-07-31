from __future__ import annotations

import re

import numpy as np

from .config import settings
from .vector_store import ScoredDocument


def rerank(query: str, candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Rerank hybrid retrieval candidates.

    Implementation:
        If `USE_CROSS_ENCODER_RERANKER=true`, the function tries to rerank with
        a Sentence Transformers `CrossEncoder`. If the model is unavailable or
        disabled, it falls back to `local_rerank`, which combines term overlap,
        retrieval score, and metadata boost.

    Usage:
        The query graph calls this after `hybrid_fusion`. It is intentionally
        configurable so the service can run production-style reranking while
        still running with no large model downloads at all.

    How it helps other functions:
        `select_by_knee_point` expects a relevance-ordered list. Better reranking
        means the knee-point cutoff and prompt augmentation receive the most
        useful chunks first.
    """
    if settings.use_cross_encoder_reranker:
        try:
            return cross_encoder_rerank(query, candidates)
        except Exception:
            return local_rerank(query, candidates)
    return local_rerank(query, candidates)


def local_rerank(query: str, candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Rerank candidates with a lightweight local lexical scorer.

    Implementation:
        The function tokenizes the query and candidate text, computes lexical
        overlap, blends it with the retrieval score, adds a small metadata boost,
        normalizes scores to 0-1, and sorts candidates descending.

    Usage:
        `rerank` calls this when cross-encoder reranking is disabled or cannot be
        loaded.

    How it helps other functions:
        It keeps the RAG pipeline fully runnable offline (no reranker download)
        while still producing ordered results for knee-point selection.
    """
    query_terms = set(_tokenize(query))
    if not candidates:
        return []

    raw_scores: list[float] = []
    for candidate in candidates:
        doc_terms = set(_tokenize(candidate.document.page_content))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        metadata_boost = 0.05 if candidate.document.metadata.get("title") else 0.0
        raw_scores.append((0.70 * overlap) + (0.25 * candidate.score) + metadata_boost)

    min_score = min(raw_scores)
    max_score = max(raw_scores)
    normalized = [
        1.0 if max_score == min_score else (score - min_score) / (max_score - min_score)
        for score in raw_scores
    ]

    reranked = [
        ScoredDocument(document=candidate.document, score=float(score), source="reranked")
        for candidate, score in zip(candidates, normalized, strict=True)
    ]
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def cross_encoder_rerank(query: str, candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Rerank candidates with a sentence-transformers cross-encoder.

    Implementation:
        The function lazily imports `CrossEncoder`, scores each query-document
        pair jointly, normalizes model scores, and returns candidates sorted by
        cross-encoder relevance.

    Usage:
        Enable it with `USE_CROSS_ENCODER_RERANKER=true`. The configured model is
        read from `CROSS_ENCODER_MODEL`.

    How it helps other functions:
        Cross-encoders examine the query and chunk together, which is usually
        more accurate than embedding similarity alone. This improves the quality
        of chunks selected by the knee-point algorithm and passed to the LLM.
    """
    if not candidates:
        return []

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.cross_encoder_model)
    pairs = [(query, candidate.document.page_content) for candidate in candidates]
    raw_scores = [float(score) for score in model.predict(pairs)]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    normalized = [
        1.0 if max_score == min_score else (score - min_score) / (max_score - min_score)
        for score in raw_scores
    ]
    reranked = [
        ScoredDocument(document=candidate.document, score=float(score), source="cross_encoder")
        for candidate, score in zip(candidates, normalized, strict=True)
    ]
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def select_by_knee_point(
    ranked: list[ScoredDocument], min_k: int = 3, max_k: int = 8
) -> list[ScoredDocument]:
    """Select a natural number of chunks using the knee-point heuristic.

    Implementation:
        Scores are normalized and treated as a curve across rank positions. The
        function measures each point's distance from the line between the first
        and last point; the farthest point is treated as the knee where relevance
        starts dropping or flattening. `min_k` and `max_k` keep the cutoff within
        practical prompt-size bounds.

    Usage:
        The query graph calls this after reranking instead of blindly passing a
        fixed `top_k` to the LLM.

    How it helps other functions:
        It controls context size for `build_augmented_prompt`, reducing noisy
        chunks while preserving enough evidence for grounded generation.
    """

    if len(ranked) <= min_k:
        return ranked

    scores = np.asarray([item.score for item in ranked], dtype=float)
    if np.allclose(scores, scores[0]):
        return ranked[: min(max_k, len(ranked))]

    x = np.linspace(0, 1, len(scores))
    y = (scores - scores.min()) / (scores.max() - scores.min())
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    distances = []

    for x_value, y_value in zip(x, y, strict=True):
        point = np.array([x_value, y_value])
        diff = start - point
        # 2D cross-product z-component (line.x*diff.y - line.y*diff.x) — the
        # scalar "perpendicular distance from the line" numerator. Computed
        # directly rather than via np.cross(), which raises ValueError on
        # 2D input on current NumPy (its implicit 2D special-case was removed).
        cross_z = line[0] * diff[1] - line[1] * diff[0]
        distance = abs(cross_z) / line_norm if line_norm else 0.0
        distances.append(distance)

    knee_index = int(np.argmax(distances)) + 1
    selected_k = max(min_k, min(knee_index, max_k, len(ranked)))
    return ranked[:selected_k]


def _tokenize(text: str) -> list[str]:
    """Tokenize text for local reranking.

    Implementation:
        Extracts alphanumeric and underscore tokens and lowercases them. This is
        deliberately the same simple style as the vector-store BM25 tokenizer.

    Usage:
        `rerank` uses it to compare query terms with candidate chunk terms.

    How it helps other functions:
        Stable lexical overlap improves rerank ordering, which improves the
        quality of chunks selected by the knee-point algorithm.
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

"""Shared context assembly (A12), applied after whichever strategy retrieved:

  dedupe -> rerank to top-n -> score cutoff -> knee cutoff -> token budget -> lost-in-the-middle

so semantic, graph and agentic answers are built from contexts selected the same way.
"""

from dataclasses import dataclass

from app.rag.context.fusion import (
    dedupe,
    fit_token_budget,
    knee_cutoff,
    lost_in_the_middle,
    score_cutoff,
)
from app.rag.retrieval.reranker import Reranker, ScoreReranker
from app.rag.strategies.base import TraceStep, traced
from app.rag.vectorstore.base import VectorSearchResult


@dataclass
class AssemblyOptions:
    top_n: int = 6
    min_relative_score: float = 0.0
    max_context_tokens: int = 6000
    reorder: bool = True
    near_duplicate: float = 0.9
    knee: bool = False  # drop results after the score knee (Kneedle)
    knee_sensitivity: float = 0.2


def _graph_context(result: VectorSearchResult) -> bool:
    """Graph fact / community contexts score on their own scale; the knee never drops them."""
    return (result.metadata.get("retriever") or "").startswith("graph")


class ContextAssembler:
    def __init__(
        self, reranker: Reranker | None = None, options: AssemblyOptions | None = None
    ) -> None:
        self._reranker = reranker or ScoreReranker()
        self._options = options or AssemblyOptions()

    async def assemble(
        self, query: str, candidates: list[VectorSearchResult], trace: list[TraceStep]
    ) -> list[VectorSearchResult]:
        options = self._options
        with traced(trace, "assemble_context", candidates=len(candidates)) as info:
            unique = dedupe(candidates, options.near_duplicate)
            ranked = await self._reranker.rerank(query, unique, options.top_n)
            above_floor = score_cutoff(ranked, min_relative=options.min_relative_score)
            kept = (
                knee_cutoff(above_floor, options.knee_sensitivity, pinned=_graph_context)
                if options.knee
                else above_floor
            )
            budgeted = fit_token_budget(kept, options.max_context_tokens)
            final = lost_in_the_middle(budgeted) if options.reorder else budgeted
            info.update(
                after_dedupe=len(unique),
                after_rerank=len(ranked),
                after_cutoff=len(above_floor),
                after_knee=len(kept) if options.knee else None,
                final=len(final),
            )
        return final

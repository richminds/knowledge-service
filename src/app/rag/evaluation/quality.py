"""Combine signals (+ optional LLM judge) into a user-facing confidence and telemetry.

Confidence rules (documented in the README so users know what the label means):
  low    - no answer was found, nothing is cited, or the judge scored groundedness or
           relevance <= 2
  high   - judged:   groundedness and answer relevance >= 4, no invalid citations
           unjudged: >= 80% of statements cite a source and no invalid citations
  medium - everything else (partially cited / partially supported)
"""

import asyncio
import random
from typing import Literal

from app.core import metrics
from app.core.config import RagMode
from app.core.logging import get_logger
from app.models.schemas import JudgeScores, QualityReport, QualitySignals
from app.rag.evaluation.judge import AnswerJudge
from app.rag.evaluation.signals import compute_signals
from app.rag.vectorstore.base import VectorSearchResult

logger = get_logger(__name__)

JudgePolicy = Literal["off", "request", "always"]


def _confidence(signals: QualitySignals, judge: JudgeScores | None) -> str:
    if not signals.answered or not signals.cited_passages:
        return "low"
    if judge is not None:
        if min(judge.groundedness, judge.answer_relevance) <= 2:
            return "low"
        if min(judge.groundedness, judge.answer_relevance) >= 4 and not signals.invalid_citations:
            return "high"
        return "medium"
    if signals.citation_coverage >= 0.8 and not signals.invalid_citations:
        return "high"
    return "medium"


def _explanation(signals: QualitySignals, judge: JudgeScores | None) -> str:
    if signals.contexts_retrieved == 0:
        return "No relevant documents were found in the knowledge base for this question."
    if not signals.answered:
        return "The retrieved documents did not contain the answer, so none was given."
    if not signals.cited_passages:
        return "The answer does not reference any of the retrieved documents; verify it."
    parts = [
        f"{round(signals.citation_coverage * 100)}% of statements cite a source; "
        f"{len(signals.cited_passages)} of {signals.contexts_retrieved} retrieved passages used."
    ]
    if signals.invalid_citations:
        parts.append(f"Some references ({signals.invalid_citations}) point to no source.")
    if judge is not None:
        parts.append(
            f"Reviewer scores (1-5): supported by sources {judge.groundedness}, "
            f"answers the question {judge.answer_relevance}, "
            f"sources relevant {judge.context_relevance}."
        )
        if judge.unsupported_claims:
            parts.append("Not supported by sources: " + "; ".join(judge.unsupported_claims[:3]))
    return " ".join(parts)


class QualityAssessor:
    def __init__(
        self,
        judge: AnswerJudge | None,
        judge_policy: JudgePolicy = "request",
        background_sample_rate: float = 0.0,
    ) -> None:
        self._judge = judge
        self._policy = judge_policy
        self._sample_rate = background_sample_rate
        self._background: set[asyncio.Task] = set()

    def should_judge(self, requested: bool) -> bool:
        if self._judge is None or self._policy == "off":
            return False
        return self._policy == "always" or requested

    async def assess(
        self,
        question: str,
        answer: str,
        contexts: list[VectorSearchResult],
        *,
        rag_mode: RagMode,
        run_judge: bool,
    ) -> QualityReport:
        signals = compute_signals(answer, len(contexts))
        judge = None
        if run_judge and self._judge is not None and signals.answered and contexts:
            judge = await self._judge.evaluate(question, answer, contexts)
        report = QualityReport(
            confidence=_confidence(signals, judge),
            explanation=_explanation(signals, judge),
            evaluated_by="signals+llm_judge" if judge else "signals",
            signals=signals,
            judge=judge,
        )
        self._record(report, rag_mode)
        return report

    def sample_in_background(
        self,
        question: str,
        answer: str,
        contexts: list[VectorSearchResult],
        *,
        rag_mode: RagMode,
        response_id: str,
    ) -> None:
        """Judge a random share of live answers after responding (telemetry only)."""
        if self._judge is None or self._policy == "off" or not contexts:
            return
        if random.random() >= self._sample_rate:  # noqa: S311 - sampling, not security
            return

        async def run() -> None:
            judge = await self._judge.evaluate(question, answer, contexts)
            if judge is None:
                return
            self._record_judge(judge, rag_mode)
            logger.info(
                "rag_quality_sampled",
                response_id=response_id,
                rag_mode=rag_mode,
                **judge.model_dump(exclude={"unsupported_claims"}),
            )

        task = asyncio.create_task(run())
        self._background.add(task)  # keep a reference until done
        task.add_done_callback(self._background.discard)

    # -- telemetry ----------------------------------------------------------------------------
    def _record(self, report: QualityReport, rag_mode: RagMode) -> None:
        attrs = {"rag.mode": str(rag_mode), "evaluator": "signals"}
        for name in ("citation_coverage", "cited_context_ratio"):
            value = getattr(report.signals, name)
            metrics.quality_score.record(value, {**attrs, "quality.metric": name})
        metrics.answers.add(
            1,
            {
                "rag.mode": str(rag_mode),
                "confidence": report.confidence,
                "answered": report.signals.answered,
            },
        )
        if report.judge is not None:
            self._record_judge(report.judge, rag_mode)

    @staticmethod
    def _record_judge(judge: JudgeScores, rag_mode: RagMode) -> None:
        attrs = {"rag.mode": str(rag_mode), "evaluator": "llm_judge"}
        for name in ("groundedness", "answer_relevance", "context_relevance"):
            metrics.quality_score.record(getattr(judge, name), {**attrs, "quality.metric": name})

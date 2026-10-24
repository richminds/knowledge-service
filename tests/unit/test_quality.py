import pytest

from app.core.config import RagMode
from app.core.usage import record_usage, track_usage
from app.models.schemas import JudgeScores
from app.rag.evaluation.judge import AnswerJudge
from app.rag.evaluation.quality import QualityAssessor, _confidence
from app.rag.evaluation.signals import compute_signals
from app.rag.generation.prompts import NO_CONTEXT_ANSWER
from app.rag.vectorstore.base import VectorSearchResult

CONTEXTS = [
    VectorSearchResult(chunk_id=f"c{i}", content=f"passage {i}", score=1.0, source=f"s{i}")
    for i in range(1, 4)
]


def _judge(groundedness: int, relevance: int) -> JudgeScores:
    return JudgeScores(groundedness=groundedness, answer_relevance=relevance, context_relevance=4)


# -- signals --------------------------------------------------------------------------------
def test_signals_measure_citation_coverage_and_usage():
    answer = (
        "Staff receive 25 days of annual leave [1]. "
        "Unused days can be carried over to next year [2][1]. "
        "Requests are approved by the line manager."
    )

    signals = compute_signals(answer, contexts_retrieved=3)

    assert signals.answered
    assert signals.cited_passages == [1, 2]
    assert signals.citation_coverage == pytest.approx(0.667, abs=0.001)
    assert signals.cited_context_ratio == pytest.approx(0.667, abs=0.001)
    assert signals.invalid_citations == []


def test_signals_flag_citations_to_missing_passages():
    signals = compute_signals("The policy changed in 2024 [7].", contexts_retrieved=3)

    assert signals.invalid_citations == [7]
    assert signals.cited_passages == []


def test_signals_detect_refusals_and_no_context():
    refusal = "I don't know - the provided passages do not cover travel insurance."

    assert not compute_signals(refusal, 3).answered
    assert not compute_signals(NO_CONTEXT_ANSWER, 0).answered


# -- confidence rules -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "judge", "expected"),
    [
        ("Staff receive twenty five days [1]. Managers approve the leave [2].", None, "high"),
        (
            "Staff receive twenty five days [1]. Managers approve every leave request.",
            None,
            "medium",
        ),
        ("Staff receive twenty five days of leave per year.", None, "low"),  # nothing cited
        ("Staff receive twenty five days [1].", _judge(5, 4), "high"),
        ("Staff receive twenty five days [1].", _judge(3, 4), "medium"),
        ("Staff receive twenty five days [1].", _judge(2, 5), "low"),
    ],
)
def test_confidence_rules(answer, judge, expected):
    assert _confidence(compute_signals(answer, 3), judge) == expected


async def test_assessor_runs_judge_only_when_policy_allows(chat_client):
    answer = "Staff receive twenty five days of leave [1]."

    on_request = QualityAssessor(AnswerJudge(chat_client), judge_policy="request")
    plain = await on_request.assess(
        "q", answer, CONTEXTS, rag_mode=RagMode.SEMANTIC, run_judge=on_request.should_judge(False)
    )
    judged = await on_request.assess(
        "q", answer, CONTEXTS, rag_mode=RagMode.SEMANTIC, run_judge=on_request.should_judge(True)
    )

    assert plain.evaluated_by == "signals" and plain.judge is None
    assert judged.evaluated_by == "signals+llm_judge"
    assert judged.judge.groundedness == 5
    assert "supported by sources 5" in judged.explanation
    assert not QualityAssessor(AnswerJudge(chat_client), judge_policy="off").should_judge(True)
    assert QualityAssessor(AnswerJudge(chat_client), judge_policy="always").should_judge(False)


async def test_judge_scores_are_clamped(chat_client):
    chat_client.judge_scores = {"groundedness": 9, "answer_relevance": "x", "context_relevance": 0}

    scores = await AnswerJudge(chat_client).evaluate("q", "a [1]", CONTEXTS)

    assert (scores.groundedness, scores.answer_relevance, scores.context_relevance) == (5, 1, 1)


# -- usage plumbing -------------------------------------------------------------------
def test_usage_tracker_accumulates_per_request():
    with track_usage() as usage:
        record_usage("answer", "gpt-4o", 100, 20)
        record_usage("quality_judge", "gpt-4o-mini", 50, 10)
    record_usage("answer", "gpt-4o", 999, 999)  # outside the request: not counted

    assert (usage.prompt_tokens, usage.completion_tokens, usage.llm_calls) == (150, 30, 2)
    assert usage.by_purpose == {"answer": 120, "quality_judge": 60}


async def test_judge_reads_long_passages_in_full():
    from app.rag.evaluation.judge import AnswerJudge
    from app.rag.vectorstore.base import VectorSearchResult

    class Capture:
        user = ""

        async def complete_json(self, system, user, *, purpose):
            Capture.user = user
            return {"groundedness": 5, "answer_relevance": 5, "context_relevance": 5}

    section = "Revision history. " * 300 + "ACN-2025-29584 changed the design."
    passage = VectorSearchResult(chunk_id="p", content=section, score=1.0, source="a.pdf")
    await AnswerJudge(Capture()).evaluate("Which ACN?", "ACN-2025-29584 [1]", [passage])

    assert "ACN-2025-29584 changed the design." in Capture.user  # beyond character 1,500


def test_citation_after_the_full_stop_counts_for_its_sentence():
    from app.rag.evaluation.signals import compute_signals

    answer = (
        "Supported file formats are:\n\n- Plain text\n- Markdown\n\n"
        "Plain text, Markdown and PDF are parsed natively; Word files use the unstructured "
        "library. [1]"
    )
    assert compute_signals(answer, 1).citation_coverage == 1.0


def test_uncited_claims_still_lower_coverage():
    from app.rag.evaluation.signals import compute_signals

    answer = "Leave is twenty five days per year. [1] Carry-over is capped at five days each year."
    assert compute_signals(answer, 1).citation_coverage == 0.5

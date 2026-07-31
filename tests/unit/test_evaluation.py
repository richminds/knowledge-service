from langchain_core.documents import Document

from rag.evaluation import evaluate_answer


def _citation(valid: bool, cited=(1,), available=(1,)) -> dict:
    return {
        "valid": valid,
        "cited_numbers": list(cited),
        "missing_citations": not cited,
        "invalid_citations": [],
        "available_citations": list(available),
    }


def test_evaluate_answer_returns_all_expected_fields():
    chunks = [Document(page_content="Refunds are issued within five business days.")]
    metrics = evaluate_answer(
        "How long do refunds take?",
        "Refunds take five business days. [1]",
        chunks,
        _citation(valid=True),
    )

    for key in (
        "faithfulness",
        "precision",
        "recall",
        "answer_relevance",
        "confidence_score",
        "confidence_level",
        "evaluator",
        "advanced_notes",
    ):
        assert key in metrics


def test_precision_and_recall_are_high_for_relevant_context():
    chunks = [
        Document(page_content="Refunds are issued within five business days of a return."),
        Document(page_content="Refund requests must be submitted through the support portal."),
    ]
    metrics = evaluate_answer(
        "How long do refunds take?",
        "Refunds take five business days. [1]",
        chunks,
        _citation(valid=True),
    )

    assert metrics["precision"] > 0.0
    assert metrics["recall"] > 0.0


def test_precision_and_recall_are_zero_with_no_chunks():
    metrics = evaluate_answer(
        "How long do refunds take?",
        "I do not have enough evidence to answer.",
        [],
        _citation(valid=False, cited=(), available=()),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_recall_drops_when_context_is_off_topic():
    on_topic = [Document(page_content="Refunds are issued within five business days.")]
    off_topic = [Document(page_content="Our warehouse is located in Chicago.")]

    on_topic_metrics = evaluate_answer(
        "How long do refunds take?", "Five business days. [1]", on_topic, _citation(valid=True)
    )
    off_topic_metrics = evaluate_answer(
        "How long do refunds take?", "Unclear. [1]", off_topic, _citation(valid=True)
    )

    assert off_topic_metrics["recall"] < on_topic_metrics["recall"]


def test_confidence_score_is_penalized_by_invalid_citations():
    chunks = [Document(page_content="Refunds are issued within five business days.")]

    valid = evaluate_answer(
        "How long do refunds take?", "Five business days. [1]", chunks, _citation(valid=True)
    )
    invalid = evaluate_answer(
        "How long do refunds take?",
        "Five business days with no citation.",
        chunks,
        _citation(valid=False, cited=(), available=(1,)),
    )

    assert invalid["confidence_score"] < valid["confidence_score"]


def test_confidence_score_is_bounded_between_zero_and_one():
    chunks = [Document(page_content="Refunds are issued within five business days.")]
    metrics = evaluate_answer(
        "How long do refunds take?", "Five business days. [1]", chunks, _citation(valid=True)
    )
    assert 0.0 <= metrics["confidence_score"] <= 1.0


def test_confidence_level_matches_score_bands():
    chunks = [Document(page_content="Refunds are issued within five business days.")]

    high = evaluate_answer(
        "refunds business days", "Five business days. [1]", chunks, _citation(valid=True)
    )
    assert high["confidence_score"] >= 0.7
    assert high["confidence_level"] == "high"

    low = evaluate_answer(
        "How long do refunds take?",
        "No relevant information found.",
        [],
        _citation(valid=False, cited=(), available=()),
    )
    assert low["confidence_score"] < 0.4
    assert low["confidence_level"] == "low"


def test_evaluator_field_reflects_configured_provider():
    metrics = evaluate_answer("q", "a", [], _citation(valid=False, cited=(), available=()))
    assert metrics["evaluator"] == "local"
    assert metrics["advanced_notes"] == "Local lexical metrics only."

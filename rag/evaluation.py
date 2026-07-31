from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.documents import Document

from .citations import CitationValidation
from .config import settings


class EvaluationMetrics(TypedDict):
    faithfulness: float
    context_precision: float
    answer_relevance: float
    evaluator: str
    advanced_notes: str


def evaluate_answer(
    question: str,
    answer: str,
    chunks: list[Document],
    citation_validation: CitationValidation,
) -> EvaluationMetrics:
    """Compute lightweight RAG quality metrics for a generated answer.

    Implementation:
        The function always computes local lexical metrics. If
        `ADVANCED_EVAL_PROVIDER` is set to `ragas`, `deepeval`, or `llm`, it also
        calls `advanced_evaluation_notes` to run or describe the configured
        production-style evaluator path.

    Usage:
        The query graph runs this after citation validation.

    How it helps other functions:
        These metrics provide immediate feedback on retrieval and generation
        quality, while the advanced evaluator hook gives a place to integrate
        RAGAS, DeepEval, or LLM-as-judge scoring.
    """
    context_text = " ".join(chunk.page_content for chunk in chunks)
    faithfulness = _faithfulness(answer, context_text, citation_validation)
    context_precision = _context_precision(question, chunks)
    answer_relevance = _term_overlap(question, answer)
    advanced_notes = advanced_evaluation_notes(question, answer, chunks)
    return {
        "faithfulness": round(faithfulness, 3),
        "context_precision": round(context_precision, 3),
        "answer_relevance": round(answer_relevance, 3),
        "evaluator": settings.advanced_eval_provider,
        "advanced_notes": advanced_notes,
    }


def advanced_evaluation_notes(question: str, answer: str, chunks: list[Document]) -> str:
    """Run or describe the configured advanced evaluation provider.

    Implementation:
        The service supports `local`, `ragas`, `deepeval`, and `llm` provider
        names. The RAGAS and DeepEval branches validate that the packages are
        importable and return integration guidance because their full scoring
        setup requires model/testset configuration. The LLM branch runs a compact
        judge prompt via the LLM Gateway.

    Usage:
        `evaluate_answer` calls this after computing local metrics.

    How it helps other functions:
        It provides a production extension point for higher-quality evaluation
        without removing the fast local metrics used during development.
    """
    provider = settings.advanced_eval_provider.lower()
    if provider == "local":
        return "Local lexical metrics only."
    if provider == "ragas":
        return _ragas_notes()
    if provider == "deepeval":
        return _deepeval_notes()
    if provider == "llm":
        return _llm_judge_notes(question, answer, chunks)
    return (
        f"Unknown advanced evaluator '{settings.advanced_eval_provider}', used local metrics only."
    )


def _ragas_notes() -> str:
    """Validate RAGAS availability and describe the integration point.

    Implementation:
        The function imports `ragas` lazily. Full RAGAS scoring requires a
        dataset object plus evaluator LLM/embedding configuration, so this helper
        reports readiness rather than forcing heavyweight setup.

    Usage:
        `advanced_evaluation_notes` calls this when
        `ADVANCED_EVAL_PROVIDER=ragas`.

    How it helps other functions:
        It lets users confirm the advanced evaluator dependency is present and
        shows where to plug in formal RAGAS metrics.
    """
    try:
        import ragas  # noqa: F401

        return (
            "RAGAS is installed. Configure a RAGAS dataset/evaluator to compute formal "
            "faithfulness and context metrics."
        )
    except ImportError:
        return "RAGAS is not installed. Install ragas or use ADVANCED_EVAL_PROVIDER=local."


def _deepeval_notes() -> str:
    """Validate DeepEval availability and describe the integration point.

    Implementation:
        The function imports `deepeval` lazily. Full DeepEval scoring usually
        requires configured test cases and metrics, so this helper reports
        readiness and keeps local evaluation lightweight.

    Usage:
        `advanced_evaluation_notes` calls this when
        `ADVANCED_EVAL_PROVIDER=deepeval`.

    How it helps other functions:
        It exposes a clean place to add DeepEval metrics without changing the
        LangGraph query flow.
    """
    try:
        import deepeval  # noqa: F401

        return "DeepEval is installed. Configure DeepEval test cases/metrics for formal scoring."
    except ImportError:
        return "DeepEval is not installed. Install deepeval or use ADVANCED_EVAL_PROVIDER=local."


def _llm_judge_notes(question: str, answer: str, chunks: list[Document]) -> str:
    """Run a compact LLM-as-judge evaluation via the LLM Gateway.

    Usage:
        `advanced_evaluation_notes` calls this when `ADVANCED_EVAL_PROVIDER=llm`.
    """
    import asyncio

    async def _judge() -> str:
        try:
            from .llm_gateway_client import get_llm_client

            context = "\n\n".join(chunk.page_content[:700] for chunk in chunks)
            prompt = (
                "Evaluate this RAG answer for faithfulness, context precision, and relevance. "
                "Return one concise sentence.\n\n"
                f"Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context}"
            )
            client = get_llm_client()
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                caller="knowledge-service.evaluation.llm_judge",
                temperature=0.0,
                max_tokens=200,
            )
            return response.content
        except Exception as exc:
            return f"LLM judge unavailable: {exc}"

    try:
        try:
            asyncio.get_running_loop()
            # Already inside an async event loop (e.g. FastAPI) — run in a
            # worker thread with its own event loop to avoid nesting.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _judge()).result(timeout=30)
        except RuntimeError:
            # No running loop — safe to run directly.
            return asyncio.run(_judge())
    except Exception as exc:
        return f"LLM judge unavailable: {exc}"


def _faithfulness(answer: str, context_text: str, citation_validation: CitationValidation) -> float:
    """Estimate whether the answer is supported by retrieved context.

    Implementation:
        The score combines citation validity and token overlap between answer and
        context. It is intentionally simple and local.

    Usage:
        `evaluate_answer` calls this as one component of its metric output.

    How it helps other functions:
        A low score can indicate that generation used weak evidence or failed to
        cite retrieved chunks properly.
    """
    citation_score = 1.0 if citation_validation["valid"] else 0.0
    overlap_score = _term_overlap(answer, context_text)
    return (0.6 * citation_score) + (0.4 * overlap_score)


def _context_precision(question: str, chunks: list[Document]) -> float:
    """Estimate how many selected chunks are relevant to the question.

    Implementation:
        The function computes lexical overlap between the question and each
        selected chunk, then reports the fraction of chunks with non-zero
        overlap.

    Usage:
        `evaluate_answer` calls this after knee-point selection and safety
        filtering.

    How it helps other functions:
        It gives feedback on retrieval, hybrid fusion, reranking, and knee-point
        cutoff quality.
    """
    if not chunks:
        return 0.0
    relevant = sum(1 for chunk in chunks if _term_overlap(question, chunk.page_content) > 0)
    return relevant / len(chunks)


def _term_overlap(left: str, right: str) -> float:
    """Compute normalized token overlap between two text values.

    Implementation:
        Text is tokenized with `_tokenize`. The score is the fraction of tokens
        from the left text that also appear in the right text.

    Usage:
        Used by all local evaluation metrics as a cheap approximation of
        relevance and support.

    How it helps other functions:
        It avoids external evaluator dependencies while still giving useful
        signals during RAG pipeline development.
    """
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for local evaluation metrics.

    Implementation:
        Extracts lowercase alphanumeric tokens and ignores punctuation.

    Usage:
        `_term_overlap` calls this for questions, answers, and context.

    How it helps other functions:
        Shared evaluation tokenization keeps faithfulness, precision, and
        relevance calculations consistent.
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

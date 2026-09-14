"""LLM layer for RAG — generation and prompt construction via the LLM Gateway.

Every LLM call in this service goes through the vendored gateway SDK's
``RemoteLLMClient`` (see ``rag/llm_gateway_client.py``), an HTTP call to a
deployed LLM Gateway instance — this package holds no provider API key and
imports no provider SDK, so there is no path that reaches a model any other
way. Routing through the gateway is what gives these calls its:
  - Fallback chain (primary → configured fallbacks)
  - Per-model retries with exponential backoff
  - Full request tracking (MongoDB or JSONL)
  - Rate limiting, cost tracking, response caching (whatever the gateway has enabled)

Two prompts leave here (build_generation_prompt):
  - the grounded RAG prompt — question + numbered context + citation rules —
    whenever retrieval selected chunks. With RAG_GENERAL_KNOWLEDGE_FALLBACK
    (default true) it lets the model answer from general knowledge, labelled
    with GENERAL_KNOWLEDGE_MARKER, when none of those chunks are relevant;
    with it false the model must refuse instead.
  - the bare question, when nothing was selected and the fallback is on, so
    the gateway routes it by its real category; the answer is labelled here.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)

# Returned verbatim when the gateway can't be reached. Deliberately says
# nothing about the retrieved evidence — the gateway is the only place an
# answer is ever composed, so a gateway outage yields no answer, not a
# locally assembled substitute.
_UNAVAILABLE_ANSWER = (
    "No answer is available right now — the language model service could not "
    "be reached. Please try again shortly."
)


# ──────────────────────────────────────────────────────── prompt builder

# Exact first line of an answer that did not come from the knowledge base.
# The model is told to emit it verbatim when nothing in the context is
# relevant; the no-context path prepends it in code. Fixed text so a client
# can detect (and style) an ungrounded answer instead of guessing from prose.
GENERAL_KNOWLEDGE_MARKER = "(Not from the knowledge base.)"

_STRICT_INSTRUCTIONS = (
    "You are a grounded RAG assistant.\n\n"
    "Answer the question using only the context below.\n"
    "If the context is insufficient, say what evidence is missing.\n"
    "Include citations like [1] or [2] for important claims.\n\n"
)

# Retrieval always hands over something — knee-point selection returns at
# least `min_k` chunks whenever the store has any — so an off-topic question
# ("what is the capital of India" against a finance knowledge base) arrives
# here with three irrelevant chunks, and under the strict instructions every
# faithful model answers "the context does not contain this". These
# instructions keep grounding and citations for what the context covers, and
# let the model answer the rest itself, labelled.
_FALLBACK_INSTRUCTIONS = (
    "You are a grounded RAG assistant.\n\n"
    "Use the context below wherever it is relevant, and cite it like [1] or "
    "[2] for every claim it supports. Never attribute to the context "
    "something it does not say.\n"
    "If nothing in the context is relevant to the question, answer from your "
    "own general knowledge instead, without citations, and begin your answer "
    f"with this exact line: {GENERAL_KNOWLEDGE_MARKER}\n\n"
)


def build_augmented_prompt(
    question: str,
    chunks: list[Document],
    *,
    general_knowledge_fallback: bool | None = None,
) -> str:
    """Build the grounded RAG prompt with numbered citation blocks.

    ``general_knowledge_fallback`` defaults to the RAG_GENERAL_KNOWLEDGE_FALLBACK
    setting: True permits a labelled general-knowledge answer when the
    context is off-topic; False is strict grounding (off-topic → refusal).
    """
    if general_knowledge_fallback is None:
        general_knowledge_fallback = settings.general_knowledge_fallback

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        title = chunk.metadata.get("title", "untitled")
        section = chunk.metadata.get("section", "unknown")
        page_number = chunk.metadata.get("page_number", "n/a")
        chunk_index = chunk.metadata.get("chunk_index", "n/a")
        context_blocks.append(
            f"[{idx}] title={title}; section={section}; page={page_number}; "
            f"source={source}; chunk={chunk_index}\n{chunk.page_content}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    instructions = _FALLBACK_INSTRUCTIONS if general_knowledge_fallback else _STRICT_INSTRUCTIONS
    return (
        f"{instructions}"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:\n"
    )


def answers_without_context(chunks: list[Document]) -> bool:
    """True when the question should go to the gateway bare, ungrounded.

    Nothing was retrieved (empty store, or every chunk removed by the
    authorization / prompt-injection filters) and the fallback is enabled.
    A grounding wrapper around an empty context can only be refused — and it
    also makes the gateway classify the prompt as "default", so the question
    never reaches the chain built for its real category (a factual lookup,
    a calculation, …). The bare question does.
    """
    return not chunks and settings.general_knowledge_fallback


def build_generation_prompt(question: str, chunks: list[Document]) -> str:
    """The exact text sent to the gateway for this question."""
    if answers_without_context(chunks):
        return question
    return build_augmented_prompt(question, chunks)


# ──────────────────────────────────────────────────────── generation

async def generate_answer(question: str, chunks: list[Document]) -> str:
    """Generate the final RAG answer via the LLM Gateway.

    The gateway (``RemoteLLMClient``) is the sole path to a model, so the call
    participates in the full gateway fallback chain and request-tracking
    pipeline. If the gateway is unreachable or every model fails, returns a
    short unavailability message — no answer is composed locally.
    """
    ungrounded = answers_without_context(chunks)
    prompt = build_generation_prompt(question, chunks)

    try:
        from .llm_gateway_client import get_llm_client

        client = get_llm_client()
        response = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            caller="knowledge-service.generate",
            temperature=settings.rag_temperature,
            max_tokens=settings.rag_max_tokens,
        )
        if ungrounded:
            # The model was never told about a knowledge base, so it cannot
            # label the answer itself — label it here, same marker.
            return f"{GENERAL_KNOWLEDGE_MARKER}\n\n{response.content}"
        return response.content

    except Exception as exc:
        logger.warning(
            "LLM Gateway unavailable for RAG generation (%s: %s) — returning "
            "unavailability message.",
            type(exc).__name__,
            exc,
        )
        return _UNAVAILABLE_ANSWER

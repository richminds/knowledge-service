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

def build_augmented_prompt(question: str, chunks: list[Document]) -> str:
    """Build the grounded RAG prompt with numbered citation blocks."""
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
    return (
        "You are a grounded RAG assistant.\n\n"
        "Answer the question using only the context below.\n"
        "If the context is insufficient, say what evidence is missing.\n"
        "Include citations like [1] or [2] for important claims.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:\n"
    )


# ──────────────────────────────────────────────────────── generation

async def generate_answer(question: str, chunks: list[Document]) -> str:
    """Generate the final RAG answer via the LLM Gateway.

    The gateway (``RemoteLLMClient``) is the sole path to a model, so the call
    participates in the full gateway fallback chain and request-tracking
    pipeline. If the gateway is unreachable or every model fails, returns a
    short unavailability message — no answer is composed locally.
    """
    prompt = build_augmented_prompt(question, chunks)

    try:
        from .llm_gateway_client import get_llm_client

        client = get_llm_client()
        response = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            caller="knowledge-service.generate",
            temperature=settings.rag_temperature,
            max_tokens=settings.rag_max_tokens,
        )
        return response.content

    except Exception as exc:
        logger.warning(
            "LLM Gateway unavailable for RAG generation (%s: %s) — returning "
            "unavailability message.",
            type(exc).__name__,
            exc,
        )
        return _UNAVAILABLE_ANSWER

"""LLM layer for RAG — generation and prompt construction via the LLM Gateway.

All LLM calls go through the vendored gateway SDK's ``RemoteLLMClient`` (see
``rag/llm_gateway_client.py``), an HTTP call to a deployed LLM Gateway
instance, so they inherit whatever that gateway provides:
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

    Routes through the gateway's RemoteLLMClient so the call participates in
    the full gateway fallback chain and request-tracking pipeline. Returns a
    local extractive summary if the gateway is unreachable or every model
    fails.
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
            "LLM Gateway unavailable for RAG generation (%s: %s) — "
            "falling back to local extractive answer.",
            type(exc).__name__,
            exc,
        )
        return _local_extractive_answer(question, chunks)


# ──────────────────────────────────────────────────────── local fallback

def _local_extractive_answer(question: str, chunks: list[Document]) -> str:
    """Return a simple evidence summary when the LLM Gateway is unavailable."""
    if not chunks:
        return "I do not have enough retrieved evidence to answer this question."

    lines = [
        "Local grounded answer (LLM Gateway unavailable):",
        "",
        f"Question: {question}",
        "",
        "Most relevant evidence:",
    ]
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("file_name") or chunk.metadata.get("source", "unknown")
        section = chunk.metadata.get("section", "unknown")
        page_number = chunk.metadata.get("page_number", "n/a")
        snippet = " ".join(chunk.page_content.split())[:500]
        lines.append(
            f"- [{idx}] {snippet}  (source: {source}, section: {section}, page: {page_number})"
        )

    lines.extend([
        "",
        "Configure RAG_GATEWAY_BASE_URL / RAG_GATEWAY_API_KEY to reach a live "
        "LLM Gateway for a fluent LLM-generated response.",
    ])
    return "\n".join(lines)

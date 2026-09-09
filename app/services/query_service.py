"""Query orchestration — runs the RAG pipeline and shapes the HTTP response."""
from __future__ import annotations

import logging

from fastapi import HTTPException, status

from rag.llm_gateway_sdk import GatewayError
from rag.retrieval import query as rag_query

from ..models.query_model import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


async def run_query(request: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline (embed → retrieve → rerank → generate)."""
    try:
        result = await rag_query(
            question=request.question,
            metadata_filter=request.metadata_filter,
            user_id=request.user_id,
            team_id=request.team_id,
            org_id=request.org_id,
            account_id=request.account_id,
        )
    except GatewayError:
        # Let app/errors.py map this to the LLM Gateway's own status code
        # (429/503/500) instead of collapsing it into an opaque 500.
        raise
    except Exception as exc:
        logger.error("RAG query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    sources = [
        {
            "source": chunk.metadata.get("source"),
            "title": chunk.metadata.get("title"),
            "section": chunk.metadata.get("section"),
            "page_number": chunk.metadata.get("page_number"),
            "chunk_id": chunk.metadata.get("chunk_id"),
            "content": chunk.page_content,
        }
        for chunk in result.get("selected_chunks", [])
    ]

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", ""),
        sources=sources,
        citation_validation=dict(result.get("citation_validation", {})),
        evaluation_metrics=dict(result.get("evaluation_metrics", {})),
    )

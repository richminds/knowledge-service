"""The query endpoint.

    POST /v1/query    ask a question via the full RAG pipeline
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import gateway_settings
from ..dependencies import get_principal, resolve_org_id, resolve_user_id
from ..models.query_model import QueryRequest, QueryResponse
from ..services import query_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question via RAG",
    responses={413: {"description": "Question too long, or too many metadata filter keys."}},
)
async def rag_query(
    body: QueryRequest, request: Request, principal: str = Depends(get_principal)
) -> QueryResponse:
    """Run the full RAG pipeline (embed → retrieve → rerank → generate)."""
    body.user_id = resolve_user_id(body.user_id, request)
    body.org_id = resolve_org_id(body.org_id, request)
    if len(body.question) > gateway_settings.max_question_length:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Question is {len(body.question)} chars, over the "
                f"{gateway_settings.max_question_length} char limit."
            ),
        )
    filter_size = len(body.metadata_filter) if body.metadata_filter else 0
    if filter_size > gateway_settings.max_metadata_filter_keys:
        raise HTTPException(
            status_code=413,
            detail=(
                f"metadata_filter has {filter_size} keys, over the "
                f"{gateway_settings.max_metadata_filter_keys} key limit."
            ),
        )
    logger.debug("Query from %s: %r", principal, body.question[:200])
    return await query_service.run_query(body)

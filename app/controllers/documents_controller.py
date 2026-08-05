"""Document and file management endpoints.

    DELETE /v1/documents/{source}    cascading delete for a source path
    GET    /v1/files                  list persisted uploaded files
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_principal
from ..models.document_model import DeleteResponse, FilesListResponse
from ..services import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["documents"])


@router.delete(
    "/documents/{source:path}",
    response_model=DeleteResponse,
    summary="Delete all chunks and parents for a source",
)
async def delete_document(source: str, principal: str = Depends(get_principal)) -> DeleteResponse:
    logger.info("Delete requested by %s for source %r", principal, source)
    return await document_service.delete_document(source)


@router.get(
    "/files",
    response_model=FilesListResponse,
    summary="List persisted uploaded files",
)
async def list_files(
    limit: int = 100,
    org_id: str | None = None,
    _: str = Depends(get_principal),
) -> FilesListResponse:
    """List persisted files. ``org_id`` scopes the list to that org (plus
    unscoped files); omit it to see only unscoped files — same fail-closed
    default as query-time chunk filtering (see rag/authorization.py)."""
    return await document_service.list_files(limit=limit, org_id=org_id)

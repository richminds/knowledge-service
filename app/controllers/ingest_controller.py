"""Ingestion endpoints.

    POST /v1/ingest             start ingestion from server-side paths
    POST /v1/upload              upload files, persist them, then ingest
    GET  /v1/ingest/{job_id}     poll ingestion job status
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from ..dependencies import get_principal, resolve_user_id
from ..models.ingest_model import IngestRequest, IngestResponse, JobStatusResponse
from ..services import ingest_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Start document ingestion from server-side paths",
)
async def start_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    principal: str = Depends(get_principal),
) -> IngestResponse:
    user_id = resolve_user_id(body.user_id, principal)
    org_id = (body.org_id or "").strip()
    logger.info(
        "Ingest requested by %s (user_id=%s, org_id=%s): %d path(s)",
        principal, user_id, org_id or "*", len(body.input_paths),
    )
    return await ingest_service.start_ingest(
        body, background_tasks, uploaded_by=user_id, org_id=org_id
    )


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=202,
    summary="Upload files and ingest them",
)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Knowledge files to ingest."),
    chunk_strategy: str = Form("recursive"),
    user_id: str | None = Form(
        default=None,
        description="End-user this upload is on behalf of; defaults to the authenticated caller.",
    ),
    org_id: str | None = Form(
        default=None,
        description=(
            "Organization this upload is scoped to; chunks are tagged with it and only "
            "returned to queries from the same org_id (see QueryRequest.org_id). Omit to "
            "leave chunks unscoped ('*', visible to every org)."
        ),
    ),
    principal: str = Depends(get_principal),
) -> IngestResponse:
    resolved_user_id = resolve_user_id(user_id, principal)
    resolved_org_id = (org_id or "").strip()
    logger.info(
        "Upload requested by %s (user_id=%s, org_id=%s): %d file(s)",
        principal, resolved_user_id, resolved_org_id or "*", len(files),
    )
    return await ingest_service.upload_and_ingest(
        files, chunk_strategy, background_tasks, uploaded_by=resolved_user_id, org_id=resolved_org_id
    )


@router.get(
    "/ingest/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_ingest_status(job_id: str, _: str = Depends(get_principal)) -> JobStatusResponse:
    return ingest_service.get_job_status(job_id)

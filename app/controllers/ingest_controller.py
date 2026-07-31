"""Ingestion endpoints.

    POST /v1/ingest             start ingestion from server-side paths
    POST /v1/upload              upload files, persist them, then ingest
    GET  /v1/ingest/{job_id}     poll ingestion job status
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from ..dependencies import get_principal
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
    logger.info("Ingest requested by %s: %d path(s)", principal, len(body.input_paths))
    return await ingest_service.start_ingest(body, background_tasks)


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
    principal: str = Depends(get_principal),
) -> IngestResponse:
    logger.info("Upload requested by %s: %d file(s)", principal, len(files))
    return await ingest_service.upload_and_ingest(files, chunk_strategy, background_tasks)


@router.get(
    "/ingest/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_ingest_status(job_id: str, _: str = Depends(get_principal)) -> JobStatusResponse:
    return ingest_service.get_job_status(job_id)

"""Ingestion orchestration — job registry, background execution, file uploads.

The job registry is a process-local dict, exactly matching the source
implementation (``backend/shared/rag/src/router.py``): status is
lost on restart and not shared across workers. ``rag/config.py`` declares
``RAG_INGESTION_JOBS_COLLECTION`` for a future MongoDB-backed registry, but it
is intentionally not wired up here — this is a faithful port, not a rewrite.
Run this service with a single worker (the default in the Dockerfile) until
that lands; scale by replica, not by worker count, exactly as the LLM Gateway
itself recommends for its own per-process state.
"""
from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, HTTPException, UploadFile, status

from rag.caller_context import caller_token_scope, get_caller_token

from rag.file_store import get_file_store
from rag.ingestion import ingest

from ..config import gateway_settings
from ..models.ingest_model import IngestRequest, IngestResponse, JobStatusResponse

logger = logging.getLogger(__name__)

# In-memory job registry — see module docstring for the multi-worker caveat.
_jobs: dict[str, dict[str, Any]] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "inserted_count": None,
        "graph_inserted_count": None,
        "error": None,
    }
    return job_id


async def start_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    uploaded_by: str = "",
    org_id: str = "",
    account_id: str = "",
) -> IngestResponse:
    """Submit an ingestion job for server-side paths. Returns immediately with a job_id to poll."""
    job_id = _new_job()
    background_tasks.add_task(
        _run_ingest_job,
        job_id=job_id,
        input_paths=request.input_paths,
        chunk_strategy=request.chunk_strategy,
        uploaded_by=uploaded_by,
        org_id=org_id,
        account_id=account_id,
        # Captured now, while the request context still exists. Background
        # tasks run after the response has unwound the middleware stack, by
        # which point the ambient token has already been reset.
        caller_token=get_caller_token(),
    )
    return IngestResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion job accepted. Poll GET /v1/ingest/{job_id} for status.",
    )


async def upload_and_ingest(
    files: list[UploadFile],
    chunk_strategy: str,
    background_tasks: BackgroundTasks,
    uploaded_by: str = "",
    org_id: str = "",
    account_id: str = "",
) -> IngestResponse:
    """Accept uploads, persist the bytes durably (GridFS), then ingest.

    Each file is first stored via the configured FileStore (GridFS by default
    — the sole backend; no S3/blob store) so it survives restarts. A transient
    working copy is then written to a temp dir and fed into the path-based
    ingestion pipeline.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files supplied.")
    if len(files) > gateway_settings.max_upload_files_per_request:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Too many files: {len(files)} > "
                f"{gateway_settings.max_upload_files_per_request}."
            ),
        )

    store = get_file_store()
    upload_dir = Path(tempfile.gettempdir()) / "knowledge-service-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    persisted = 0
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        if not data:
            continue
        if len(data) > gateway_settings.max_upload_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"{f.filename!r} is {len(data)} bytes, over the "
                    f"{gateway_settings.max_upload_file_size_mb} MB limit."
                ),
            )
        name = Path(f.filename).name  # strip any path components

        # 1) Persist durably (MongoDB GridFS by default).
        try:
            await store.save(
                name, data,
                content_type=f.content_type,
                metadata={
                    "origin": "upload",
                    "chunk_strategy": chunk_strategy,
                    "uploaded_by": uploaded_by,
                    "account_id": account_id,
                },
            )
            persisted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Upload: failed to persist %r to %s store: %s", name, store.backend, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist {name!r} to {store.backend} store: {exc}",
            ) from exc

        # 2) Materialise a transient working copy for the path-based ingester.
        dest = upload_dir / name
        dest.write_bytes(data)
        saved_paths.append(str(dest))

    if not saved_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded files had no readable content.",
        )

    job_id = _new_job()
    background_tasks.add_task(
        _run_ingest_job,
        job_id=job_id,
        input_paths=saved_paths,
        chunk_strategy=chunk_strategy,
        uploaded_by=uploaded_by,
        org_id=org_id,
        account_id=account_id,
        # Captured now, while the request context still exists. Background
        # tasks run after the response has unwound the middleware stack, by
        # which point the ambient token has already been reset.
        caller_token=get_caller_token(),
    )
    return IngestResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"Persisted {persisted} file(s) to the {store.backend} store and queued "
            f"ingestion. Poll GET /v1/ingest/{job_id} for status."
        ),
    )


def get_job_status(job_id: str) -> JobStatusResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found."
        )
    return JobStatusResponse(job_id=job_id, **job)


async def _run_ingest_job(
    job_id: str,
    input_paths: list[str],
    chunk_strategy: str,
    uploaded_by: str = "",
    org_id: str = "",
    account_id: str = "",
    caller_token: str = "",
) -> None:
    """Execute ingestion in the background and update the job registry.

    ``caller_token`` is re-bound for the life of the job because embedding
    every chunk goes out through the API Gateway, which bills and budgets the
    call against the user who submitted the upload (see
    rag/llm_gateway_client.py). It is a snapshot taken when the job was
    queued: a long ingest that outlives the token's expiry will start failing
    its embedding calls, and the job is marked failed like any other error.
    Shorten documents or lengthen AUTH_ACCESS_TTL_MINUTES if that bites.
    """
    _jobs[job_id]["status"] = "running"
    try:
        with caller_token_scope(caller_token):
            result = await ingest(
                input_paths=input_paths,
                chunk_strategy=chunk_strategy,
                uploaded_by=uploaded_by,
                org_id=org_id,
                account_id=account_id,
            )  # type: ignore[arg-type]
        _jobs[job_id].update(
            {
                "status": "completed",
                "inserted_count": result.get("inserted_count", 0),
                "graph_inserted_count": result.get("graph_inserted_count", 0),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.info(
            "Ingestion job %s completed: %d chunk(s) inserted.",
            job_id,
            result.get("inserted_count", 0),
        )
    except Exception as exc:
        _jobs[job_id].update(
            {
                "status": "failed",
                "error": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.error("Ingestion job %s failed: %s", job_id, exc, exc_info=True)

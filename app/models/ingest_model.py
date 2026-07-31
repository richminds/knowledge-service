"""Request/response schemas for the ingestion endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    input_paths: list[str] = Field(
        ...,
        min_length=1,
        description="File or directory paths to ingest, as seen by this service's process.",
        examples=[["/data/docs/contract.pdf"]],
    )
    chunk_strategy: Literal["recursive", "semantic"] = Field(
        default="recursive",
        description="'recursive' (default, fast) or 'semantic' (embedding-based, costs more).",
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "End-user this ingestion is on behalf of, recorded as `uploaded_by` on every "
            "resulting chunk's metadata. Defaults to the authenticated caller (principal) "
            "when omitted — mirrors QueryRequest.user_id, the retrieval-side equivalent."
        ),
    )


class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "completed" | "failed"
    inserted_count: int | None = None
    graph_inserted_count: int | None = None
    error: str | None = None
    started_at: str
    completed_at: str | None = None

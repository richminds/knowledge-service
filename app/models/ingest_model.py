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
    org_id: str | None = Field(
        default=None,
        description=(
            "Organization this ingestion is scoped to, recorded as `org_id` on every "
            "resulting chunk's metadata. Unlike user_id, this IS an access-control field — "
            "at query time, a chunk tagged with a real org_id is only returned to queries "
            "from that same org_id (see QueryRequest.org_id). Omit to leave chunks unscoped "
            "('*', visible to every org, matching today's default)."
        ),
    )
    account_id: str | None = Field(
        default=None,
        description=(
            "Application (auth-service app account) this ingestion is scoped to, recorded "
            "as `account_id` on every resulting chunk's metadata and enforced at query time "
            "exactly like org_id. Omit to leave chunks unscoped ('*')."
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

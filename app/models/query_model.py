"""Request/response schemas for the query endpoint."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="The question to answer using the RAG pipeline."
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters applied at retrieval time (e.g. {'source': '...'}).",
    )
    user_id: str | None = Field(default=None, description="Applies authorized_users filtering.")
    team_id: str | None = Field(default=None, description="Applies authorized_teams filtering.")
    org_id: str | None = Field(
        default=None,
        description=(
            "Restricts results to chunks tagged with this org_id at ingest time; "
            "unscoped ('*') chunks are always included. This is a hard tenant boundary, "
            "not an OR-list like user_id/team_id — a chunk tagged with a different org_id "
            "is never returned, regardless of user_id/team_id."
        ),
    )
    account_id: str | None = Field(
        default=None,
        description=(
            "Restricts results to chunks ingested under this application (an "
            "auth-service app account); unscoped ('*') chunks are always included. "
            "A hard boundary like org_id — for a JWT caller it comes from the token's "
            "account_id claim, so it reflects the account chosen at sign-in."
        ),
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    evaluation_metrics: dict[str, Any]

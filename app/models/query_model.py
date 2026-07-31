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


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    evaluation_metrics: dict[str, Any]

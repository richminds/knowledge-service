"""Request/response schemas for operational visibility endpoints (stats, config)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StatsResponse(BaseModel):
    total_chunks: int
    collection: str
    db: str
    file_store_backend: str
    graph_retrieval_enabled: bool
    graph_db_backend: str
    web_search_enabled: bool
    gateway_base_url: str


class ConfigResponse(BaseModel):
    service: dict[str, Any]
    rag: dict[str, Any]

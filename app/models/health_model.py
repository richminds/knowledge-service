"""Response schemas for the health endpoints."""
from __future__ import annotations

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    name: str
    status: str  # "ok" | "degraded" | "unavailable" | "disabled"
    detail: str


class LivenessResponse(BaseModel):
    status: str = "alive"
    version: str


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "degraded" | "not_ready"
    version: str
    dependencies: list[DependencyStatus]

"""Operational visibility endpoints — admin role required.

    GET /v1/stats     indexed-chunk count + active backend/config snapshot
    GET /v1/config    resolved, non-secret configuration

Mirrors the LLM Gateway's own admin-gated /v1/stats and /v1/config.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..config import gateway_settings
from ..dependencies import require_admin
from ..models.stats_model import ConfigResponse, StatsResponse
from ..services import stats_service

router = APIRouter(prefix="/v1", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats", response_model=StatsResponse, summary="Knowledge base stats")
async def get_stats() -> StatsResponse:
    return await stats_service.get_stats()


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Resolved (non-secret) configuration",
)
async def get_config() -> ConfigResponse:
    if not gateway_settings.expose_config_endpoint:
        raise HTTPException(
            status_code=404,
            detail="Config endpoint disabled (KNOWLEDGE_EXPOSE_CONFIG_ENDPOINT=false).",
        )
    return stats_service.get_config()

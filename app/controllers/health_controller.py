"""Health endpoints — always public, no credential required.

    GET /health         readiness (alias of /health/ready)
    GET /health/live     liveness — process is up, no I/O
    GET /health/ready    readiness — MongoDB + LLM Gateway reachability

Liveness must never touch a dependency: a Mongo or gateway blip should not get
the container killed. Readiness may, because that is exactly the signal a
load balancer needs.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from rag.config import settings as rag_settings
from rag.llm_gateway_client import get_llm_client
from rag.mongo_connection import get_connection

from ..models.health_model import DependencyStatus, LivenessResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse, summary="Liveness probe")
async def live(request: Request) -> LivenessResponse:
    return LivenessResponse(version=request.app.version)


async def _readiness(request: Request) -> ReadinessResponse:
    deps: list[DependencyStatus] = []

    if rag_settings.mongo_uri:
        try:
            conn = await get_connection(
                uri=rag_settings.mongo_uri, db_name=rag_settings.mongo_db_name
            )
            reachable = await conn.ping()
            deps.append(
                DependencyStatus(
                    name="mongodb",
                    status="ok" if reachable else "degraded",
                    detail=(
                        f"db={rag_settings.mongo_db_name}"
                        if reachable
                        else "Unreachable — ingestion and retrieval cannot serve."
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            deps.append(DependencyStatus(name="mongodb", status="degraded", detail=str(exc)))
    else:
        deps.append(
            DependencyStatus(
                name="mongodb", status="unavailable", detail="RAG_MONGO_URI is not set."
            )
        )

    try:
        health = await get_llm_client().health()
        gw_status = str(health.get("status", "ready"))
        deps.append(
            DependencyStatus(
                name="llm_gateway",
                status="ok" if gw_status in ("ready", "ok") else "degraded",
                detail=f"{rag_settings.gateway_base_url} → {gw_status}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        deps.append(
            DependencyStatus(
                name="llm_gateway",
                status="degraded",
                detail=f"{rag_settings.gateway_base_url} unreachable: {exc}",
            )
        )

    mongo_ok = any(d.name == "mongodb" and d.status == "ok" for d in deps)
    if not mongo_ok:
        overall = "not_ready"
    elif any(d.status == "degraded" for d in deps):
        overall = "degraded"
    else:
        overall = "ready"

    return ReadinessResponse(status=overall, version=request.app.version, dependencies=deps)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(request: Request, response: Response) -> ReadinessResponse:
    result = await _readiness(request)
    if result.status == "not_ready":
        response.status_code = 503
    return result


@router.get("", response_model=ReadinessResponse, summary="Health (alias of /health/ready)")
async def health(request: Request, response: Response) -> ReadinessResponse:
    return await ready(request, response)

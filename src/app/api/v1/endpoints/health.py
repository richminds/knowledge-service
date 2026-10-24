from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.deps import SettingsDep
from app.core.logging import get_logger
from app.rag.graph.factory import get_graph_store
from app.rag.vectorstore.factory import get_vector_store, vector_stores

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe (config + every enabled store)")
async def readiness(settings: SettingsDep) -> JSONResponse:
    checks: dict[str, str] = {}
    errors = settings.configuration_errors()
    checks["config"] = "ok" if not errors else "; ".join(errors)

    stores = []
    if settings.vector_enabled and settings.vector_store_configured:
        # One check per backend, so a down store is named instead of hiding behind the other.
        backends = vector_stores(settings)
        for provider, store in backends.items():
            label = f"vector_store:{provider}" if len(backends) > 1 else "vector_store"
            stores.append((label, lambda _s, store=store: store))
    elif settings.vector_enabled:
        stores.append(("vector_store", get_vector_store))
    if settings.graph_enabled:
        stores.append(("graph_store", get_graph_store))
    for name, factory in stores:
        try:
            await factory(settings).ping()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            logger.warning("readiness_failed", check=name, error=str(exc))
            checks[name] = "unavailable"

    ready = all(value == "ok" for value in checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ready else "not_ready",
            "rag_mode": settings.rag_mode,
            "enabled_modes": settings.rag_enabled_modes,
            "checks": checks,
        },
    )

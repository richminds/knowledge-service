"""Knowledge Service — FastAPI application.

A standalone, generic RAG service extracted from Portless's
``backend/shared/rag`` so it can be reused by Portless and any other
application, over HTTP, instead of being embedded in one codebase. Every LLM
and embedding call is made through a deployed LLM Gateway instance — this
service holds no provider credential and imports no provider SDK (see
``rag/llm_gateway_client.py``).

Startup ensures MongoDB indexes exist (vector search index, regular indexes,
graph-store indexes when enabled) — best-effort, matching the source
implementation: an index-creation hiccup degrades the relevant retrieval leg
rather than crashing the service. Logging is configured first, before
anything else, so every subsequent startup line is already structured (see
``app/logging_config.py``).

Run it::

    uvicorn app.main:app --reload --port 8090
    # or: python run.py
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag import __version__
from rag.config import settings as rag_settings
from rag.graph_store import ensure_graph_indexes
from rag.llm_gateway_client import aclose_llm_client
from rag.mongo_connection import close_all as close_mongo
from rag.vector_store import ensure_indexes

from .config import gateway_settings
from .controllers import (
    admin_controller,
    documents_controller,
    health_controller,
    ingest_controller,
    query_controller,
)
from .errors import register_exception_handlers
from .logging_config import configure_logging
from .middleware.auth import AuthMiddleware
from .middleware.request_context import RequestContextMiddleware

logger = logging.getLogger(__name__)


def _warn_on_open_access() -> None:
    """Refuse to let an unauthenticated deployment go unnoticed.

    Open access is the right default for localhost and CI, but on a deployed
    instance it means anyone who finds the URL can read/write the knowledge
    base and spend the LLM Gateway's quota on this service's behalf.
    """
    if gateway_settings.parsed_api_keys() or rag_settings.auth_enabled:
        return
    message = (
        "AUTH IS DISABLED — every endpoint is open. Set KNOWLEDGE_API_KEYS "
        "(recommended for service-to-service) or RAG_AUTH_ENABLED=true + "
        "RAG_JWT_SECRET before exposing this service."
    )
    if gateway_settings.is_production:
        logger.error("!!! %s", message)
    else:
        logger.warning(message)


async def _ensure_mongo_indexes() -> None:
    """Best-effort index creation — never blocks startup on a Mongo hiccup."""
    if not rag_settings.mongo_uri:
        logger.warning(
            "RAG_MONGO_URI is not set — ingestion and retrieval will fail until "
            "it is configured. The service stays up so /health reports why."
        )
        return
    try:
        await ensure_indexes()
        logger.info("Knowledge Service: MongoDB indexes ensured.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Knowledge Service startup index check failed: %s", exc)

    try:
        await ensure_graph_indexes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Knowledge Service graph-store index check failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(gateway_settings.log_level, gateway_settings.log_format)
    _warn_on_open_access()

    await _ensure_mongo_indexes()

    logger.info(
        "Knowledge Service ready — gateway=%s graph_retrieval=%s web_search=%s",
        rag_settings.gateway_base_url,
        rag_settings.graph_retrieval_enabled,
        rag_settings.web_search_enabled,
    )

    yield

    await aclose_llm_client()
    await close_mongo()
    logger.info("Knowledge Service shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Service",
        version=__version__,
        description=(
            "Generic RAG service: document ingestion (Markdown/text/PDF, "
            "recursive or semantic chunking, parent-child context) and retrieval "
            "(hybrid semantic + keyword + optional graph + optional web search, "
            "reranking, knee-point selection, prompt-injection screening, cited "
            "generation) over MongoDB Atlas. Every LLM/embedding call is made "
            "through a configured LLM Gateway instance — this service holds no "
            "provider credential of its own."
        ),
        lifespan=lifespan,
        root_path=gateway_settings.root_path,
        docs_url="/docs" if gateway_settings.docs_enabled else None,
        redoc_url="/redoc" if gateway_settings.docs_enabled else None,
        openapi_url="/openapi.json" if gateway_settings.docs_enabled else None,
    )

    # Middleware runs bottom-up: request context wraps auth, so a 401 still
    # gets a request ID and an access-log line.
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestContextMiddleware)

    origins = gateway_settings.parsed_cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)

    app.include_router(health_controller.router)
    app.include_router(ingest_controller.router)
    app.include_router(query_controller.router)
    app.include_router(documents_controller.router)
    app.include_router(admin_controller.router)

    @app.get("/", tags=["health"], summary="Service banner")
    async def root() -> dict:
        return {
            "service": "knowledge-service",
            "version": __version__,
            "docs": "/docs" if gateway_settings.docs_enabled else None,
            "health": "/health",
        }

    return app


app = create_app()

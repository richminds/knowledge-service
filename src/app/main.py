from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.v1.router import api_router
from app.core.caller_context import CallerTokenMiddleware
from app.core.config import get_settings
from app.core.exceptions import (
    RagAgentError,
    rag_agent_exception_handler,
    unhandled_exception_handler,
)
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.mongo import close_mongo_clients
from app.rag.generation.llm_gateway import aclose_llm_client
from app.rag.graph.factory import get_graph_store
from app.rag.vectorstore.factory import get_vector_store, vector_stores

settings = get_settings()
configure_logging(settings.log_level, mask_pii=settings.log_pii_masking)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app_startup",
        environment=settings.app_env,
        rag_mode=settings.rag_mode,
        enabled_modes=settings.rag_enabled_modes,
        auth_mode=settings.auth_mode,
    )
    for error in settings.configuration_errors():
        # Don't crash-loop the container; /health/ready reports these until fixed.
        logger.error("configuration_error", detail=error)

    stores = []
    if settings.vector_enabled and settings.vector_store_configured:
        stores.append(get_vector_store(settings))
        for provider, vector_store in vector_stores(settings).items():
            if settings.mongodb_create_indexes_on_startup:
                try:
                    await vector_store.ensure_index()
                except Exception:  # noqa: BLE001
                    logger.exception("ensure_index_failed", provider=provider)
    if settings.graph_enabled and settings.graph_store_configured:
        graph_store = get_graph_store(settings)
        stores.append(graph_store)
        # MONGODB_CREATE_INDEXES_ON_STARTUP also gates the Mongo graph's collections.
        if settings.graph_create_schema_on_startup and (
            settings.graph_store_provider == "neo4j" or settings.mongodb_create_indexes_on_startup
        ):
            try:
                await graph_store.ensure_schema()
            except Exception:  # noqa: BLE001
                logger.exception("ensure_graph_schema_failed")
    yield
    for store in stores:
        await store.close()
    await aclose_llm_client()
    await close_mongo_clients()
    logger.info("app_shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    # Hide interactive docs in production; keep them for local/QA.
    docs_url=None if settings.app_env == "prod" else "/docs",
    redoc_url=None if settings.app_env == "prod" else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)
# Carries the caller's bearer token to the LLM gateway calls made while serving the request.
app.add_middleware(CallerTokenMiddleware)

app.add_exception_handler(RagAgentError, rag_agent_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(api_router, prefix=settings.api_v1_prefix)

if settings.ui_enabled:
    _CHAT_PAGE = Path(__file__).parent / "static" / "chat.html"

    @app.get("/ui", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        # The page holds no data; it calls the authenticated API with the user's credential.
        return FileResponse(_CHAT_PAGE, media_type="text/html")

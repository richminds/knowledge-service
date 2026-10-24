from fastapi import APIRouter, Depends

from app.api.v1.endpoints import (
    admin,
    chat,
    conversations,
    feedback,
    health,
    ingestion,
    retrieval,
)
from app.core.auth import get_current_user, require_admin_role
from app.core.ratelimit import enforce_rate_limit

api_router = APIRouter()
# Health probes stay unauthenticated so App Service / load balancers can reach them.
api_router.include_router(health.router, tags=["health"])

authenticated = [Depends(get_current_user)]
# Query endpoints are also rate limited per user / API key (C-197).
limited = [Depends(get_current_user), Depends(enforce_rate_limit)]

api_router.include_router(
    ingestion.router, prefix="/ingestion", tags=["ingestion"], dependencies=authenticated
)
# Serves /retrieval and /rag/modes.
api_router.include_router(retrieval.router, tags=["retrieval"], dependencies=limited)
api_router.include_router(chat.router, prefix="/chat", tags=["chat"], dependencies=limited)
api_router.include_router(
    feedback.router, prefix="/feedback", tags=["feedback"], dependencies=authenticated
)
api_router.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["conversations"],
    dependencies=authenticated,
)
# Admin RAG studio: role-gated and rate limited (each run makes several LLM calls).
api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_role), Depends(enforce_rate_limit)],
)

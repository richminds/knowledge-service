from fastapi import APIRouter

from app.api.deps import RetrievalServiceDep, SettingsDep
from app.core.auth import CurrentUser
from app.models.schemas import RagModesResponse, RetrievalRequest, RetrievalResponse

router = APIRouter()


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_chunks(
    request: RetrievalRequest,
    user: CurrentUser,
    retrieval_service: RetrievalServiceDep,
    settings: SettingsDep,
) -> RetrievalResponse:
    """Contexts a RAG mode would ground an answer on (after ACL filtering, guardrails and
    context assembly), without LLM generation."""
    if request.include_trace and not settings.rag_allow_trace:
        request = request.model_copy(update={"include_trace": False})
    return await retrieval_service.retrieve(
        request, user, request.top_k or settings.retrieval_top_k
    )


@router.get("/rag/modes", response_model=RagModesResponse)
async def rag_modes(retrieval_service: RetrievalServiceDep) -> RagModesResponse:
    """Which RAG strategies and vector stores this deployment serves, and whether `rag_mode` /
    `vector_store` can be overridden per request."""
    registry = retrieval_service.registry
    return RagModesResponse(
        default=registry.default_mode,
        enabled=registry.enabled_modes,
        allow_override=registry.allow_override,
        vector_stores=list(retrieval_service.vector_backends),
        default_vector_store=retrieval_service.default_vector_backend,
    )

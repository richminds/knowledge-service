import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import RetrievalServiceDep, SettingsDep
from app.core.auth import CurrentUser
from app.core.exceptions import RagAgentError
from app.core.logging import get_logger
from app.core.progress import STEP_LABELS, progress
from app.core.usage import track_usage
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()
logger = get_logger(__name__)


def _effective(request: ChatRequest, settings) -> ChatRequest:
    if request.include_trace and not settings.rag_allow_trace:
        return request.model_copy(update={"include_trace": False})
    return request


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: CurrentUser,
    retrieval_service: RetrievalServiceDep,
    settings: SettingsDep,
) -> ChatResponse:
    """RAG answer with citations, a quality report (confidence + explanation), token usage
    and latency. Set `evaluate: true` for LLM-judge scores. With a `conversation_id` and no
    `history`, previous turns are loaded from the conversation store."""
    request = _effective(request, settings)
    return await retrieval_service.chat(request, user, request.top_k or settings.retrieval_top_k)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_stream(
    request: ChatRequest,
    user: CurrentUser,
    retrieval_service: RetrievalServiceDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Server-Sent Events: `step`* (live progress: "Searching documents", "Checking the
    knowledge graph"...), [`trace`], `citations`, `token`*, then `quality` (confidence,
    explanation, which citations were used, usage, and `withheld` + `replacement` when the
    output guardrail rejects the streamed text) and `done` - or `error`."""
    request = _effective(request, settings)
    steps: asyncio.Queue = asyncio.Queue()

    def on_step(step: str, detail: dict) -> None:
        if step in STEP_LABELS:
            steps.put_nowait(
                {"step": step, "label": STEP_LABELS[step], "duration_ms": detail.get("duration_ms")}
            )

    async def prepare():
        with progress(on_step):
            return await retrieval_service.chat_stream(
                request, user, request.top_k or settings.retrieval_top_k
            )

    task = asyncio.create_task(prepare())

    async def events() -> AsyncIterator[str]:
        while not task.done() or not steps.empty():  # A-42: stream progress while retrieving
            try:
                yield _sse("step", await asyncio.wait_for(steps.get(), 0.25))
            except TimeoutError:
                continue
        try:
            result = task.result()
        except RagAgentError as exc:
            yield _sse("error", {"detail": exc.message})
            return
        if request.include_trace:
            yield _sse("trace", [step.model_dump() for step in result.trace])
        yield _sse("citations", [c.model_dump() for c in result.citations])
        answer: list[str] = []
        try:
            with track_usage(result.usage):
                async for token in result.tokens:
                    answer.append(token)
                    yield _sse("token", token)
            summary = await retrieval_service.finish_stream(
                result, "".join(answer), evaluate=request.evaluate
            )
        except RagAgentError as exc:
            logger.warning("chat_stream_failed", error=exc.message)
            yield _sse("error", {"detail": exc.message})
            return
        yield _sse(
            "quality",
            {
                "answer_type": summary.answer_type,
                "quality": summary.quality.model_dump() if summary.quality else None,
                "citations": [c.model_dump() for c in summary.citations],
                "usage": summary.usage.model_dump(),
                "latency_ms": summary.latency_ms,
                "withheld": summary.withheld,
                "replacement": summary.replacement,
            },
        )
        yield _sse(
            "done",
            {
                "response_id": result.response_id,
                "conversation_id": result.prepared.conversation_id,
                "rag_mode": result.rag_mode,
                "vector_store": result.prepared.vector_store,
                "cached": result.cached is not None,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

"""Admin RAG studio (role-gated): what the deployment can do, and scored test runs that
compare RAG modes, vector stores and answer models on fresh (uncached) answers."""

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import RetrievalServiceDep, SettingsDep, get_chat_client
from app.core.audit import audit
from app.core.auth import AdminUser
from app.core.config import Settings
from app.core.logging import get_logger
from app.models.schemas import (
    AdminEvalRequest,
    AdminEvalResponse,
    AdminGoldenCase,
    AdminRagConfig,
    ChatRequest,
    RagModesResponse,
    VectorStoreStatus,
    VectorStoreStatusResponse,
)
from app.rag.evaluation.harness import (
    GoldenCase,
    check_thresholds,
    response_metrics,
    score_correctness,
)
from app.rag.generation.llm_client import ChatClient, answer_model
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.factory import vector_stores

router = APIRouter()
logger = get_logger(__name__)


def _thresholds(path: str) -> dict[str, dict[str, float]]:
    file = Path(path)
    if not file.is_file():
        return {}
    rules = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return {mode: {k: float(v) for k, v in (r or {}).items()} for mode, r in rules.items()}


def _golden_set(path: str) -> list[AdminGoldenCase]:
    file = Path(path)
    if not file.is_file():
        return []
    cases = []
    for i, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            cases.append(AdminGoldenCase.model_validate({"id": f"case-{i}", **data}))
        except ValueError:
            logger.warning("golden_case_invalid", line=i)
    return cases


def _models(settings: Settings) -> dict[str, str]:
    models = {
        "answer": settings.azure_openai_chat_deployment,
        "utility": settings.utility_deployment,
        "embedding": f"{settings.azure_openai_embedding_deployment} "
        f"({settings.embedding_dimensions}d)",
        "judge": settings.utility_deployment if settings.quality_judge != "off" else "off",
        "reranker": settings.reranker_model if settings.reranker == "cohere" else settings.reranker,
        "fallback": settings.azure_openai_fallback_chat_deployment,
        "vision": settings.azure_openai_vision_deployment,
        "whisper": settings.azure_openai_whisper_deployment,
    }
    return {k: v for k, v in models.items() if v}


@router.get("/rag/config", response_model=AdminRagConfig)
async def rag_config(
    user: AdminUser, settings: SettingsDep, retrieval_service: RetrievalServiceDep
) -> AdminRagConfig:
    registry = retrieval_service.registry
    return AdminRagConfig(
        modes=RagModesResponse(
            default=registry.default_mode,
            enabled=registry.enabled_modes,
            allow_override=registry.allow_override,
            vector_stores=list(retrieval_service.vector_backends),
            default_vector_store=retrieval_service.default_vector_backend,
        ),
        answer_models=settings.answer_models,
        default_answer_model=settings.azure_openai_chat_deployment,
        models=_models(settings),
        retrieval={
            "top_k": settings.retrieval_top_k,
            "rerank_top_n": settings.retrieval_rerank_top_n,
            "min_score": settings.retrieval_min_score,
            "search_mode": settings.retrieval_mode,
            "reranker": settings.reranker,
            "knee_cutoff": settings.retrieval_knee_cutoff,
            "graph_knee_cutoff": settings.graph_knee_cutoff,
            "max_context_tokens": settings.context_max_tokens,
            "graph_search_mode": settings.graph_search_mode,
            "escalation": settings.rag_escalation_enabled,
        },
        quality_judge=settings.quality_judge,
        ingestion={
            "max_upload_mb": settings.ingestion_max_upload_mb,
            "default_targets": [str(t) for t in settings.ingestion_targets],
            "graph_enabled": settings.graph_enabled,
            "supported_extensions": settings.ingestion_upload_extensions,
        },
        thresholds=_thresholds(settings.eval_thresholds_path),
        golden_set=_golden_set(settings.eval_golden_set_path),
    )


async def _store_status(name: str, store: VectorStore, collection: str, default: str):
    started = time.perf_counter()
    try:
        count: int | None = await store.count(collection)
        error = ""
    except Exception as exc:  # noqa: BLE001 - reported per store, never fails the page
        count, error = None, str(exc) or type(exc).__name__
    return VectorStoreStatus(
        name=name,
        default=name == default,
        count=count,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        error=error,
    )


@router.get("/rag/vector-stores", response_model=VectorStoreStatusResponse)
async def vector_store_status(
    user: AdminUser,
    settings: SettingsDep,
    retrieval_service: RetrievalServiceDep,
    collection: Annotated[str, Query(pattern=r"^[A-Za-z0-9_\-]{1,64}$")] = "default",
) -> VectorStoreStatusResponse:
    """Record count of `collection` in every vector store. Comparing answer quality between
    stores is only fair when they hold the same content (backfill: ingest_cli.py --force)."""
    stores = vector_stores(settings) if settings.vector_enabled else {}
    default = retrieval_service.default_vector_backend or settings.vector_store_provider
    statuses = list(
        await asyncio.gather(
            *(_store_status(name, store, collection, default) for name, store in stores.items())
        )
    )
    counts = {s.count for s in statuses}
    return VectorStoreStatusResponse(
        collection=collection,
        stores=statuses,
        in_sync=bool(statuses) and None not in counts and len(counts) == 1,
    )


@router.post("/rag/evaluate", response_model=AdminEvalResponse)
async def evaluate(
    request: AdminEvalRequest,
    user: AdminUser,
    settings: SettingsDep,
    retrieval_service: RetrievalServiceDep,
    chat_client: Annotated[ChatClient, Depends(get_chat_client)],
) -> AdminEvalResponse:
    """One scored RAG run: answer, citations, trace, quality report and evaluation metrics.

    Bypasses the answer cache (a cached answer would hide the model / mode under test) and
    honours `rag_mode` even when RAG_ALLOW_MODE_OVERRIDE is off, so admins can compare.
    """
    model = request.model or settings.azure_openai_chat_deployment
    if model not in settings.answer_models:
        raise HTTPException(
            422,
            f"Model '{model}' is not selectable (RAG_SELECTABLE_MODELS)",
        )
    top_k = request.top_k or settings.retrieval_top_k
    chat_request = ChatRequest(
        query=request.query,
        collection=request.collection,
        top_k=top_k,
        rag_mode=request.rag_mode,
        vector_store=request.vector_store,
        include_trace=True,
        evaluate=request.judge,
        filters=request.filters,
    )
    started = time.perf_counter()
    with answer_model(model):
        response = await retrieval_service.chat(
            chat_request, user, top_k, use_cache=False, force_mode=True
        )
    latency_ms = (time.perf_counter() - started) * 1000

    case = GoldenCase(
        question=request.query,
        collection=request.collection,
        expected_sources=request.expected_sources,
        expected_answer=request.expected_answer,
    )
    metrics = response_metrics(case, response, top_k, request.rag_mode, latency_ms)
    if request.rag_mode is not None:
        metrics.pop("routing_accuracy", None)  # no expected mode to route to
    correctness = await score_correctness(chat_client, case, response.answer)
    if correctness is not None:
        metrics["correctness"] = correctness

    audit(
        "admin_rag_evaluate",
        user_id=user.user_id,
        response_id=response.response_id,
        rag_mode=str(response.rag_mode),
        model=model,
    )
    return AdminEvalResponse(
        model=model,
        top_k=top_k,
        response=response,
        metrics={k: round(v, 4) for k, v in metrics.items()},
        threshold_violations=check_thresholds(
            {"summary": {str(response.rag_mode): metrics}}, settings.eval_thresholds_path
        ),
    )

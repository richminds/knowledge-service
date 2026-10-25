"""Dual-store ingestion and per-request retrieval through the real service stack."""

import pytest

from app.core.config import RagMode
from app.core.exceptions import VectorBackendError
from app.models.schemas import ChatRequest, RetrievalRequest
from app.rag.ingestion.chunking import ChunkingOptions
from app.rag.ingestion.governance import GovernancePolicy
from app.rag.ingestion.pipeline import (
    IngestionPipeline,
    LoadedDocument,
    PipelineOptions,
)
from app.rag.query.transforms import QueryTransformer
from app.rag.retrieval.retriever import VectorRetriever
from app.rag.strategies.registry import RagStrategyRegistry
from app.rag.strategies.semantic import SemanticRagStrategy
from app.rag.vectorstore.multi import MultiVectorStore, set_vector_backend
from app.services.retrieval_service import RetrievalService
from tests.fakes import FakeVectorStore

LEAVE = LoadedDocument(
    source="leave.md", text="Annual leave policy: staff receive 25 days of leave per year."
)


@pytest.fixture
def stores() -> dict[str, FakeVectorStore]:
    return {"mongo": FakeVectorStore(), "mongo_b": FakeVectorStore()}


@pytest.fixture
def multi(stores) -> MultiVectorStore:
    set_vector_backend(None)
    return MultiVectorStore(dict(stores), default="mongo")


@pytest.fixture
def service(multi, embedding_client, chat_client) -> RetrievalService:
    semantic = SemanticRagStrategy(
        VectorRetriever(embedding_client, multi), multi, transformer=QueryTransformer(chat_client)
    )
    return RetrievalService(
        RagStrategyRegistry({RagMode.SEMANTIC: semantic}, RagMode.SEMANTIC, allow_override=False),
        chat_client,
        vector_backends=multi.names,
        default_vector_backend="mongo",
    )


async def _ingest(multi, embedding_client):
    pipeline = IngestionPipeline(
        embedding_client,
        multi,
        options=PipelineOptions(
            chunking=ChunkingOptions(chunk_size=50, chunk_overlap=10, min_chunk_tokens=5),
            governance=GovernancePolicy(min_text_chars=1),
            default_targets=("vector",),
        ),
    )
    return await pipeline.run_for_documents(
        [LEAVE], "default", targets=["vector"], metadata={"account_id": "acct-test"}
    )


async def test_ingestion_fills_both_stores(multi, stores, embedding_client):
    stats = await _ingest(multi, embedding_client)

    assert stats.chunk_count > 0
    assert set(stores["mongo"].records) == set(stores["mongo_b"].records) != set()


async def test_each_request_is_served_by_the_store_it_names(
    service, multi, stores, embedding_client
):
    await _ingest(multi, embedding_client)
    # Make the stores distinguishable: only mongo_b knows about the extra source.
    for record in list(stores["mongo_b"].records.values()):
        stores["mongo"].records.pop(record.chunk_id)
    default = await service.retrieve(RetrievalRequest(query="annual leave days"), top_k=3)
    assert default.vector_store == "mongo"
    assert default.results == []

    request = RetrievalRequest(query="annual leave days", vector_store="mongo_b")
    with pytest.raises(VectorBackendError, match="disabled"):  # allow_override=False
        await service.retrieve(request, top_k=3)


async def test_admin_runs_may_override_even_when_user_overrides_are_off(
    service, multi, stores, embedding_client
):
    await _ingest(multi, embedding_client)
    for chunk_id in list(stores["mongo"].records):
        stores["mongo"].records.pop(chunk_id)

    response = await service.chat(
        ChatRequest(query="annual leave days", vector_store="mongo_b"),
        top_k=3,
        use_cache=False,
        force_mode=True,
    )

    assert response.vector_store == "mongo_b"
    assert response.citations, "mongo_b holds the content, so it must have answered"


async def test_unknown_vector_store_is_rejected(service):
    with pytest.raises(VectorBackendError, match="not enabled"):
        await service.retrieve(RetrievalRequest(query="x", vector_store="qdrant"), top_k=3)


async def test_eval_runner_scores_each_store_separately(
    service, multi, stores, embedding_client, chat_client
):
    from app.rag.evaluation.harness import EvalRunner, GoldenCase

    await _ingest(multi, embedding_client)
    for chunk_id in list(stores["mongo"].records):
        stores["mongo"].records.pop(chunk_id)
    case = GoldenCase("How many annual leave days?", "a", expected_sources=["leave.md"])

    report = await EvalRunner(service, top_k=3).run(
        [case], [RagMode.SEMANTIC], vector_stores=["mongo", "mongo_b"]
    )

    summary = report["summary"]
    assert set(summary) == {"semantic@mongo", "semantic@mongo_b"}
    assert summary["semantic@mongo_b"]["hit"] == 1.0
    assert summary["semantic@mongo"].get("hit", 0.0) == 0.0
    assert summary["semantic@mongo_b"]["retrieval_ms"] >= 0

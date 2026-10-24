import pytest

from app.core.config import RagMode
from app.core.exceptions import IngestionConfigError, RagModeError
from app.rag.ingestion.pipeline import IngestionPipeline, LoadedDocument
from app.rag.strategies.registry import RagStrategyRegistry
from app.rag.types import RetrievalFilters

DEFAULT = RetrievalFilters(collection="default")

ORG_DOC = LoadedDocument(
    source="org.md",
    text="Alice manages the Finance department. Finance approves Travel requests.",
)
LEAVE_DOC = LoadedDocument(
    source="leave.md", text="Annual leave policy: staff receive 25 days of leave per year."
)


async def _ingest_both(ingestion_pipeline):
    await ingestion_pipeline.run_for_documents(
        [ORG_DOC, LEAVE_DOC], "default", targets=["vector", "graph"]
    )


# -- ingestion targets ------------------------------------------------------------------
async def test_graph_target_extracts_entities_and_relations(ingestion_pipeline, graph_store):
    stats = await ingestion_pipeline.run_for_documents([ORG_DOC], "default", targets=["graph"])

    assert stats.targets == ["graph"]
    assert stats.chunk_count == 0  # vector store untouched
    assert stats.entity_count >= 3
    assert stats.relation_count >= 2
    assert {e.key for c in graph_store.chunks.values() for e in c.entities} >= {
        "alice",
        "finance",
        "travel",
    }


async def test_reingesting_rebuilds_the_documents_subgraph(ingestion_pipeline, graph_store):
    await ingestion_pipeline.run_for_documents([ORG_DOC], "default", targets=["graph"])
    edited = LoadedDocument(source="org.md", text="Bob runs Procurement.")
    await ingestion_pipeline.run_for_documents([edited], "default", targets=["graph"])

    keys = {e.key for c in graph_store.chunks.values() for e in c.entities}
    assert keys == {"bob", "procurement"}


async def test_unconfigured_target_fails_fast(embedding_client, vector_store):
    pipeline = IngestionPipeline(embedding_client, vector_store)  # no graph store

    with pytest.raises(IngestionConfigError):
        await pipeline.run_for_documents([ORG_DOC], "default", targets=["graph"])


# -- strategies ---------------------------------------------------------------------------
async def test_semantic_strategy_returns_vector_contexts(ingestion_pipeline, strategies):
    await _ingest_both(ingestion_pipeline)

    outcome = await strategies[RagMode.SEMANTIC].retrieve("annual leave days", DEFAULT, 3)

    assert outcome.contexts[0].source == "leave.md"
    assert outcome.contexts[0].metadata["retriever"] == "vector"
    assert [s.step for s in outcome.trace] == ["vector_search"]


async def test_graph_strategy_returns_facts_then_linked_chunks(ingestion_pipeline, strategies):
    await _ingest_both(ingestion_pipeline)

    outcome = await strategies[RagMode.GRAPH].retrieve("Who runs Finance?", DEFAULT, 3)

    facts, *chunks = outcome.contexts
    assert facts.source == "knowledge-graph"
    assert "Alice -[RELATED_TO]-> Finance" in facts.content
    assert chunks[0].source == "org.md"
    steps = [s.step for s in outcome.trace]
    assert steps[:3] == ["query_entities", "entity_lookup", "graph_mode"]
    assert outcome.trace[2].detail["mode"] == "local"
    assert {"graph_traversal", "linked_chunks"} <= set(steps)


async def test_graph_strategy_with_no_matching_entities_returns_nothing(strategies):
    outcome = await strategies[RagMode.GRAPH].retrieve("Who runs Finance?", DEFAULT, 3)

    assert outcome.contexts == []


async def test_agentic_strategy_routes_to_both_tools_and_stops_when_satisfied(
    ingestion_pipeline, strategies, chat_client
):
    await _ingest_both(ingestion_pipeline)

    outcome = await strategies[RagMode.AGENTIC].retrieve("Who runs Finance?", DEFAULT, 4)

    steps = [s.step for s in outcome.trace]
    # enough evidence on the first pass -> generate -> verified
    assert steps == ["route", "retrieve", "grade", "sufficiency", "generate", "answer_check"]
    assert outcome.trace[1].detail["tools"] == ["semantic_search", "graph_local_search"]
    retrievers = {c.metadata.get("retriever") for c in outcome.contexts}
    assert retrievers == {"vector", "graph"}
    assert outcome.answer and not outcome.review_required


async def test_agentic_strategy_rewrites_when_grader_rejects_everything(
    ingestion_pipeline, strategies, chat_client
):
    await _ingest_both(ingestion_pipeline)
    chat_client.grade_keep = set()  # grader finds nothing relevant

    outcome = await strategies[RagMode.AGENTIC].retrieve("Who runs Finance?", DEFAULT, 4)

    steps = [s.step for s in outcome.trace]
    assert steps.count("rewrite") == 2  # max_iterations=3 -> 2 rewrites, then give up
    rewrite = next(s for s in outcome.trace if s.step == "rewrite")
    assert rewrite.detail["query"] == "Who runs Finance? details"
    assert outcome.contexts == [] and outcome.answer is None


# -- registry (the mode switch) -------------------------------------------------------------
def test_registry_uses_default_and_honours_request_override(strategies):
    registry = RagStrategyRegistry(strategies, RagMode.SEMANTIC)

    assert registry.resolve(None).mode == RagMode.SEMANTIC
    assert registry.resolve(RagMode.GRAPH).mode == RagMode.GRAPH


def test_registry_rejects_disabled_mode(strategies):
    registry = RagStrategyRegistry(
        {RagMode.SEMANTIC: strategies[RagMode.SEMANTIC]}, RagMode.SEMANTIC
    )

    with pytest.raises(RagModeError, match="not enabled"):
        registry.resolve(RagMode.AGENTIC)


def test_registry_rejects_override_when_disabled(strategies):
    registry = RagStrategyRegistry(strategies, RagMode.SEMANTIC, allow_override=False)

    assert registry.resolve(RagMode.SEMANTIC).mode == RagMode.SEMANTIC
    with pytest.raises(RagModeError, match="disabled"):
        registry.resolve(RagMode.GRAPH)


# -- unified router (I-01) and escalation (I-03) -------------------------------------------------
def test_auto_routes_by_complexity_with_fallback(strategies):
    registry = RagStrategyRegistry(strategies, RagMode.AUTO)
    assert registry.resolve(None, "simple").mode == RagMode.HYBRID
    assert registry.resolve(None, "relationship").mode == RagMode.GRAPH
    assert registry.resolve(None, "global").mode == RagMode.GRAPH
    assert registry.resolve(RagMode.AUTO, "multi_step").mode == RagMode.AGENTIC
    semantic_only = RagStrategyRegistry(
        {RagMode.SEMANTIC: strategies[RagMode.SEMANTIC]}, RagMode.AUTO
    )
    assert semantic_only.resolve(None, "simple").mode == RagMode.SEMANTIC
    assert semantic_only.resolve(None, "multi_step").mode == RagMode.SEMANTIC
    assert semantic_only.escalate(RagMode.SEMANTIC) is None
    assert registry.escalate(RagMode.SEMANTIC).mode == RagMode.HYBRID
    assert registry.escalate(RagMode.HYBRID).mode == RagMode.HYBRID_GRAPH
    assert registry.escalate(RagMode.GRAPH).mode == RagMode.HYBRID_GRAPH
    # graph-only is never a step up from hybrid_graph
    assert registry.escalate(RagMode.HYBRID_GRAPH).mode == RagMode.AGENTIC
    no_graph = RagStrategyRegistry(
        {m: strategies[m] for m in (RagMode.SEMANTIC, RagMode.HYBRID)}, RagMode.AUTO
    )
    assert no_graph.resolve(None, "relationship").mode == RagMode.HYBRID
    assert no_graph.escalate(RagMode.HYBRID) is None


async def test_service_routes_and_escalates_low_confidence(
    ingestion_pipeline, strategies, chat_client
):
    from app.models.schemas import ChatRequest
    from app.rag.query.analysis import QueryAnalyzer
    from app.services.retrieval_service import RetrievalService

    await _ingest_both(ingestion_pipeline)
    chat_client.analysis = {
        "standalone_question": "Who runs Finance?",
        "intent": "rag",
        "complexity": "relationship",
        "filters": {},
    }
    service = RetrievalService(
        RagStrategyRegistry(strategies, RagMode.AUTO),
        chat_client,
        analyzer=QueryAnalyzer(chat_client, doc_types=[]),
        escalation=True,
    )
    routed = await service.chat(ChatRequest(query="Who runs Finance?", include_trace=True))
    assert routed.rag_mode == RagMode.GRAPH
    assert any(s.step == "router" and s.detail["mode"] == "graph" for s in routed.trace)

    chat_client.analysis["complexity"] = "simple"
    chat_client.answer_template = "I don't know."  # uncited -> low confidence
    escalated = await service.chat(ChatRequest(query="Who runs Finance?", include_trace=True))
    steps = [s for s in escalated.trace if s.step == "escalation"]
    assert steps and steps[0].detail == {"from": "hybrid", "to": "hybrid_graph"}

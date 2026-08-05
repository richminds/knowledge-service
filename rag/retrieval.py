"""LangGraph orchestration for the query/retrieval pipeline.

Separated from ``ingestion.py`` (the write-side pipeline) because the two are
independent concerns — see that module's docstring for the rationale.

Pipeline:
  embed_query → semantic_search ──┐
                keyword_search ──┤
                  graph_search ──┤→ hybrid_fusion → parent_child_expand
                    web_search ──┘
  → rerank → knee_point_select → safety_filter
  → augment → generate → validate_citations → evaluate → END

safety_filter (prompt-injection screening) runs on the full merged chunk
list after knee-point selection, so web-search chunks are screened exactly
like internal chunks before anything reaches augment/generate.

Fully async so MongoDB and LLM Gateway calls never block the FastAPI event
loop.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .authorization import filter_authorized_results
from .citations import CitationValidation, validate_citations
from .config import settings
from .embeddings import get_embeddings
from .evaluation import EvaluationMetrics, evaluate_answer
from .graph_store import GraphStore
from .llm import build_augmented_prompt, generate_answer
from .rerank import rerank, select_by_knee_point
from .security import afilter_prompt_injection_chunks
from .vector_store import ScoredDocument, VectorStore, expand_parent_context, hybrid_fusion
from .web_search import WebSearchStore

logger = logging.getLogger(__name__)


class QueryState(TypedDict, total=False):
    question: str
    metadata_filter: dict[str, Any]
    user_id: str
    team_id: str
    org_id: str
    query_embedding: list[float]
    semantic_results: list[ScoredDocument]
    keyword_results: list[ScoredDocument]
    graph_results: list[ScoredDocument]
    web_results: list[ScoredDocument]
    hybrid_chunks: list[ScoredDocument]
    parent_chunks: list[ScoredDocument]
    reranked_chunks: list[ScoredDocument]
    selected_chunks: list[Document]
    augmented_prompt: str
    answer: str
    citation_validation: CitationValidation
    evaluation_metrics: EvaluationMetrics


def build_query_graph():
    """Compile the async LangGraph query pipeline."""
    graph = StateGraph(QueryState)
    graph.add_node("embed_query", _embed_query)
    graph.add_node("semantic_search", _semantic_search)
    graph.add_node("keyword_search", _keyword_search)
    graph.add_node("graph_search", _graph_search)
    graph.add_node("web_search", _web_search)
    graph.add_node("hybrid_fusion", _fuse_results)
    graph.add_node("parent_child_expand", _parent_child_expand)
    graph.add_node("rerank", _rerank_results)
    graph.add_node("knee_point_select", _knee_point_select)
    graph.add_node("safety_filter", _safety_filter)
    graph.add_node("augment", _augment)
    graph.add_node("generate", _generate)
    graph.add_node("validate_citations", _validate_citations)
    graph.add_node("evaluate", _evaluate)

    graph.set_entry_point("embed_query")
    graph.add_edge("embed_query", "semantic_search")
    graph.add_edge("semantic_search", "keyword_search")
    graph.add_edge("keyword_search", "graph_search")
    graph.add_edge("graph_search", "web_search")
    graph.add_edge("web_search", "hybrid_fusion")
    graph.add_edge("hybrid_fusion", "parent_child_expand")
    graph.add_edge("parent_child_expand", "rerank")
    graph.add_edge("rerank", "knee_point_select")
    graph.add_edge("knee_point_select", "safety_filter")
    graph.add_edge("safety_filter", "augment")
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", "validate_citations")
    graph.add_edge("validate_citations", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()


# ──────────────────────────────────────────────── nodes

async def _embed_query(state: QueryState) -> QueryState:
    """Embed the user question for Atlas $vectorSearch."""
    embeddings = get_embeddings()
    query_embedding = await embeddings.aembed_query(state["question"])
    return {"query_embedding": query_embedding}


async def _semantic_search(state: QueryState) -> QueryState:
    """Run Atlas $vectorSearch and apply authorization filtering."""
    vector_store = VectorStore(get_embeddings())
    results = await vector_store.semantic_search(
        state["question"],
        top_k=settings.retrieval_top_k,
        metadata_filter=state.get("metadata_filter") or None,
    )
    return {
        "semantic_results": filter_authorized_results(
            results,
            user_id=state.get("user_id"),
            team_id=state.get("team_id"),
            org_id=state.get("org_id"),
        )
    }


async def _keyword_search(state: QueryState) -> QueryState:
    """Run in-memory BM25 over a metadata-filtered MongoDB document fetch."""
    vector_store = VectorStore(get_embeddings())
    results = await vector_store.keyword_search(
        state["question"],
        top_k=settings.retrieval_top_k,
        metadata_filter=state.get("metadata_filter") or None,
    )
    return {
        "keyword_results": filter_authorized_results(
            results,
            user_id=state.get("user_id"),
            team_id=state.get("team_id"),
            org_id=state.get("org_id"),
        )
    }


async def _graph_search(state: QueryState) -> QueryState:
    """Retrieve relationship-aware candidates from the graph store (no-op unless enabled)."""
    results = await GraphStore().retrieve(
        state["question"],
        top_k=settings.graph_retrieval_top_k,
        metadata_filter=state.get("metadata_filter") or None,
    )
    return {
        "graph_results": filter_authorized_results(
            results,
            user_id=state.get("user_id"),
            team_id=state.get("team_id"),
            org_id=state.get("org_id"),
        )
    }


async def _web_search(state: QueryState) -> QueryState:
    """Fetch free web search results (Tavily) as a fourth retrieval leg.

    No-op (empty result) when RAG_WEB_SEARCH_ENABLED is false or
    TAVILY_API_KEY is unset — see web_search.py for why this fails open
    rather than returning a stub.
    """
    if not settings.web_search_enabled:
        return {"web_results": []}
    results = await WebSearchStore().retrieve(state["question"], top_k=settings.web_search_top_k)
    return {
        "web_results": filter_authorized_results(
            results,
            user_id=state.get("user_id"),
            team_id=state.get("team_id"),
            org_id=state.get("org_id"),
        )
    }


async def _fuse_results(state: QueryState) -> QueryState:
    """Merge semantic, keyword, graph, and web candidates with Reciprocal Rank Fusion."""
    fused = hybrid_fusion(
        state.get("semantic_results", []),
        state.get("keyword_results", []),
        top_k=settings.retrieval_top_k * 2,
        graph_results=state.get("graph_results"),
        web_results=state.get("web_results"),
    )
    return {"hybrid_chunks": fused}


async def _parent_child_expand(state: QueryState) -> QueryState:
    """Replace child chunks with broader parent context from MongoDB."""
    expanded = await expand_parent_context(state.get("hybrid_chunks", []))
    return {"parent_chunks": expanded}


async def _rerank_results(state: QueryState) -> QueryState:
    """Reorder parent-expanded candidates with the configured reranker."""
    reranked = rerank(state["question"], state.get("parent_chunks", []))
    return {"reranked_chunks": reranked}


async def _knee_point_select(state: QueryState) -> QueryState:
    """Choose the final context window using the knee-point heuristic."""
    selected = select_by_knee_point(state.get("reranked_chunks", []))
    return {"selected_chunks": [item.document for item in selected]}


async def _safety_filter(state: QueryState) -> QueryState:
    """Remove chunks that contain prompt-injection indicators."""
    safe = await afilter_prompt_injection_chunks(state.get("selected_chunks", []))
    return {"selected_chunks": safe}


async def _augment(state: QueryState) -> QueryState:
    """Build the evidence-grounded prompt with citation numbers."""
    augmented_prompt = build_augmented_prompt(
        state["question"],
        state.get("selected_chunks", []),
    )
    return {"augmented_prompt": augmented_prompt}


async def _generate(state: QueryState) -> QueryState:
    """Generate the final answer via the LLM Gateway."""
    answer = await generate_answer(
        state["question"],
        state.get("selected_chunks", []),
    )
    return {"answer": answer}


async def _validate_citations(state: QueryState) -> QueryState:
    """Validate citation markers in the generated answer against selected chunks."""
    citation_validation = validate_citations(
        state.get("answer", ""),
        state.get("selected_chunks", []),
    )
    return {"citation_validation": citation_validation}


async def _evaluate(state: QueryState) -> QueryState:
    """Compute lightweight RAG quality metrics for the answer."""
    evaluation_metrics = evaluate_answer(
        state["question"],
        state.get("answer", ""),
        state.get("selected_chunks", []),
        state.get(
            "citation_validation",
            {
                "valid": False,
                "cited_numbers": [],
                "missing_citations": True,
                "invalid_citations": [],
                "available_citations": [],
            },
        ),
    )
    return {"evaluation_metrics": evaluation_metrics}


# ──────────────────────────────────────────────── public convenience API

async def query(
    question: str,
    metadata_filter: dict[str, Any] | None = None,
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
) -> QueryState:
    """Run the full query pipeline and return the final state."""
    graph = build_query_graph()
    initial: QueryState = {
        "question": question,
        "metadata_filter": metadata_filter or {},
        "user_id": user_id or "",
        "team_id": team_id or "",
        "org_id": org_id or "",
    }
    return await graph.ainvoke(initial)

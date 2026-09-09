"""LangGraph orchestration for the ingestion pipeline.

Separated from ``retrieval.py`` (the query/retrieval pipeline) because the two
are independent concerns with different scaling and failure characteristics:
ingestion is a write-heavy, background-job workload (see
``app/services/ingest_service.py``) while retrieval is a read-heavy, latency
-sensitive request path. Keeping them in separate modules means either can
move to its own worker/process later without touching the other.

Pipeline:
  parse_and_load → chunk → embed_and_insert → graph_insert → END

Fully async so MongoDB and LLM Gateway calls never block the FastAPI event
loop.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from starlette.concurrency import run_in_threadpool

from .chunking import ChunkStrategy, chunk_documents
from .embeddings import get_embeddings
from .graph_store import GraphStore
from .loader import load_documents
from .parent_store import MongoParentStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionState(TypedDict, total=False):
    input_paths: list[str]
    chunk_strategy: ChunkStrategy
    # Who triggered this ingestion — an end-user ID supplied by the calling
    # application, or its own principal as a fallback. Recorded as
    # `uploaded_by` on every document/chunk's metadata (see loader.py) purely
    # for provenance; it does not affect the "*" (public) authorization
    # default set at the same time.
    uploaded_by: str
    # Organization this ingestion is scoped to. Unlike uploaded_by, this IS
    # an access-control field — recorded as `metadata["org_id"]` and enforced
    # as a hard tenant boundary at query time (see rag/authorization.py).
    org_id: str
    # Application (auth-service app account) this ingestion was made under.
    # Same kind of access-control field as org_id: recorded as
    # `metadata["account_id"]` and enforced as a hard boundary at query time.
    account_id: str
    raw_documents: list[Document]
    chunks: list[Document]
    inserted_count: int
    graph_inserted_count: int


def build_ingestion_graph():
    """Compile the async LangGraph ingestion pipeline."""
    graph = StateGraph(IngestionState)
    graph.add_node("parse_and_load", _parse_and_load)
    graph.add_node("chunk", _chunk)
    graph.add_node("embed_and_insert", _embed_and_insert)
    graph.add_node("graph_insert", _graph_insert)
    graph.set_entry_point("parse_and_load")
    graph.add_edge("parse_and_load", "chunk")
    graph.add_edge("chunk", "embed_and_insert")
    graph.add_edge("embed_and_insert", "graph_insert")
    graph.add_edge("graph_insert", END)
    return graph.compile()


# ──────────────────────────────────────────────── nodes

async def _parse_and_load(state: IngestionState) -> IngestionState:
    """Load raw files from input_paths into LangChain Documents."""
    raw_paths = state.get("input_paths") or []
    paths = [Path(p) for p in raw_paths] if raw_paths else []
    raw_documents = load_documents(
        paths,
        uploaded_by=state.get("uploaded_by") or None,
        org_id=state.get("org_id") or None,
        account_id=state.get("account_id") or None,
    )
    logger.info("parse_and_load: loaded %d document(s).", len(raw_documents))
    return {"raw_documents": raw_documents}


async def _chunk(state: IngestionState) -> IngestionState:
    """Split documents into child chunks and persist parent context to MongoDB.

    ``chunk_documents`` is a plain (non-async) function whose semantic
    strategy makes a blocking embedding call to the LLM Gateway — run it in
    the threadpool so a semantic-strategy ingestion never stalls the event
    loop, mirroring how the LLM Gateway itself pushes its own blocking
    embedding calls off the loop (see its app/services/embedding_service.py).
    """
    embeddings = get_embeddings()
    strategy: ChunkStrategy = state.get("chunk_strategy", "recursive")
    chunks, parent_docs = await run_in_threadpool(
        chunk_documents, state.get("raw_documents", []), embeddings, strategy
    )

    # Persist parent docs asynchronously before chunks are embedded
    if parent_docs:
        await MongoParentStore().save(parent_docs)
        logger.debug("chunk: saved %d parent doc(s) to MongoDB.", len(parent_docs))

    logger.info("chunk: produced %d child chunk(s).", len(chunks))
    return {"chunks": chunks}


async def _embed_and_insert(state: IngestionState) -> IngestionState:
    """Embed child chunks and upsert them into the Atlas vector collection."""
    embeddings = get_embeddings()
    vector_store = VectorStore(embeddings)
    inserted_count = await vector_store.upsert(state.get("chunks", []))
    logger.info("embed_and_insert: %d chunk(s) written to MongoDB Atlas.", inserted_count)
    return {"inserted_count": inserted_count}


async def _graph_insert(state: IngestionState) -> IngestionState:
    """Insert chunk nodes and relationships into the graph store (no-op unless enabled)."""
    graph_store = GraphStore()
    graph_inserted_count = await graph_store.upsert_chunks(state.get("chunks", []))
    return {"graph_inserted_count": graph_inserted_count}


# ──────────────────────────────────────────────── public convenience API

async def ingest(
    input_paths: list[str],
    chunk_strategy: ChunkStrategy = "recursive",
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> IngestionState:
    """Run the full ingestion pipeline and return the final state."""
    graph = build_ingestion_graph()
    return await graph.ainvoke(
        {
            "input_paths": input_paths,
            "chunk_strategy": chunk_strategy,
            "uploaded_by": uploaded_by or "",
            "org_id": org_id or "",
            "account_id": account_id or "",
        }
    )

"""Graph store for relationship-aware (hybrid) RAG retrieval.

Backends (select via ``RAG_GRAPH_DB_BACKEND``):
  • ``mongo``    — implemented here (MongoStore below); same shared MongoDB
    database as everything else in the service, no new connection/credentials.
  • ``neo4j``    — implemented here (Cypher over the async Neo4j driver).
  • ``arangodb`` / ``falkordb`` — reserved; fall back to the no-op store until wired.

Activation:
  Set ``RAG_GRAPH_RETRIEVAL_ENABLED=true`` and ``RAG_GRAPH_DB_BACKEND=mongo``
  (the recommended default — no extra service to run) or ``neo4j`` (+
  ``RAG_GRAPH_DB_URI`` / ``USER`` / ``PASSWORD``). When disabled — or when the
  backend/credentials are missing, or a required driver isn't installed — the
  store degrades to a safe no-op that returns empty results, so the vector +
  keyword + RRF pipeline keeps working unchanged. ``ingestion.py`` /
  ``retrieval.py`` never have to change: they always talk to the
  ``GraphStore`` wrapper, which delegates to whichever backend resolved at
  import time.

Graph schema (identical shape across backends — Mongo just stores it as
documents with array fields instead of separate nodes/edges):
  (:Document {source, title})
  (:Chunk    {chunk_id, content, source, section, page_number})
  (:Entity   {name})
  (Chunk)-[:PART_OF]->(Document)
  (Chunk)-[:MENTIONS]->(Entity)
  (Chunk)-[:ADJACENT_TO]->(Chunk)   -- consecutive chunks of the same source

Entity extraction is a lightweight, dependency-free heuristic (capitalised tokens
+ numeric codes minus stopwords), shared by every backend. Swap
``_extract_entities`` for spaCy / an LLM NER call for higher precision — no
other code needs to change.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document

from .config import settings
from .vector_store import ScoredDocument

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────── entity extraction

# Capitalised words (≥3 chars) and numeric codes (HS codes, ids). Deliberately
# simple and deterministic; replace with real NER for production-grade recall.
_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z]{2,})\b")
_CODE_RE = re.compile(r"\b(\d{4,10})\b")
_STOPWORDS = {
    "the", "and", "for", "this", "that", "these", "those", "with", "from",
    "our", "your", "their", "please", "thank", "thanks", "dear", "best",
    "regards", "hello", "hi", "we", "you", "they", "it", "as", "is", "are",
    "was", "were", "will", "would", "can", "could", "should", "may", "might",
    "subject", "date", "note", "however", "therefore", "per", "each", "all",
}


def _extract_entities(text: str, max_entities: int = 25) -> list[str]:
    """Extract normalised entity strings (lowercased) from ``text``.

    Deterministic so the same phrase in a chunk and in a query maps to the same
    ``Entity`` node — that overlap is what graph retrieval matches on.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        ent = m.group(1).lower()
        if ent in _STOPWORDS or ent in seen:
            continue
        seen.add(ent)
        out.append(ent)
    for m in _CODE_RE.finditer(text):
        code = m.group(1)
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out[:max_entities]


def _chunk_fields(chunk: Document) -> dict[str, Any]:
    """Pull the graph fields out of a LangChain Document (defensive on metadata)."""
    md = chunk.metadata or {}
    source = str(md.get("source", "") or "")
    chunk_id = str(
        md.get("chunk_id") or f"{source}:{abs(hash(chunk.page_content)) & 0xFFFFFFFF:08x}"
    )
    return {
        "chunk_id": chunk_id,
        "source": source,
        "content": chunk.page_content or "",
        "section": str(md.get("section", "") or ""),
        "page": md.get("page_number", md.get("page")),
        "title": str(md.get("title", "") or source),
        "entities": _extract_entities(chunk.page_content or ""),
    }


# ─────────────────────────────────────────────────── Cypher statements

_UPSERT_CHUNKS_CYPHER = """
UNWIND $rows AS row
MERGE (d:Document {source: row.source})
  ON CREATE SET d.title = row.title
MERGE (c:Chunk {chunk_id: row.chunk_id})
  SET c.content = row.content, c.section = row.section,
      c.page_number = row.page, c.source = row.source
MERGE (c)-[:PART_OF]->(d)
"""

_UPSERT_ENTITIES_CYPHER = """
UNWIND $rows AS row
UNWIND row.entities AS ent
  MATCH (c:Chunk {chunk_id: row.chunk_id})
  MERGE (e:Entity {name: ent})
  MERGE (c)-[:MENTIONS]->(e)
"""

_ADJ_CYPHER = """
UNWIND $pairs AS p
MATCH (a:Chunk {chunk_id: p.a}), (b:Chunk {chunk_id: p.b})
MERGE (a)-[:ADJACENT_TO]->(b)
"""

_RETRIEVE_CYPHER = """
UNWIND $entities AS ent
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: ent})
WHERE $source IS NULL OR c.source = $source
WITH c, count(DISTINCT e) AS shared
RETURN c.chunk_id AS chunk_id, c.content AS content, c.source AS source,
       c.section AS section, c.page_number AS page, shared AS shared
ORDER BY shared DESC
LIMIT $top_k
"""

_DELETE_CYPHER = """
MATCH (c:Chunk {source: $source})
DETACH DELETE c
"""


# ─────────────────────────────────────────────────── no-op backend

class NoopGraphStore:
    """Safe no-op used when graph retrieval is disabled or unconfigured."""

    def __init__(self, warn: bool = False) -> None:
        self._warn = warn

    async def upsert_chunks(self, chunks: list[Document]) -> int:
        if self._warn:
            logger.warning(
                "GraphStore.upsert_chunks: no graph backend wired — skipping graph ingestion."
            )
        return 0

    async def retrieve(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ScoredDocument]:
        if self._warn:
            logger.warning(
                "GraphStore.retrieve: no graph backend wired — returning empty graph results."
            )
        return []

    async def delete_by_source(self, source: str) -> int:
        return 0


# ─────────────────────────────────────────────────── Mongo backend

_GRAPH_INDEXES = [
    {"keys": [("chunk_id", 1)], "options": {"unique": True, "name": "graph_chunk_id_unique"}},
    {"keys": [("entities", 1)], "options": {"name": "graph_entities_idx"}},
    {"keys": [("source", 1)], "options": {"name": "graph_source_idx"}},
]


async def ensure_graph_indexes() -> None:
    """Create the Mongo graph store's indexes. Safe/idempotent to call on every
    startup; a no-op unless graph retrieval is enabled with backend=mongo."""
    if not settings.graph_retrieval_enabled or (settings.graph_db_backend or "").lower() != "mongo":
        return
    from .mongo_connection import get_connection

    conn = await get_connection(uri=settings.mongo_uri, db_name=settings.mongo_db_name)
    await conn.ensure_indexes(settings.graph_chunks_collection, _GRAPH_INDEXES)


class MongoGraphStore:
    """Relationship-aware store backed by the service's shared MongoDB database.

    Same graph shape as Neo4jGraphStore (PART_OF / MENTIONS / ADJACENT_TO), but
    modeled as plain documents with array fields rather than separate nodes and
    edges — the single-hop "shared entity count" retrieval this pipeline
    actually does is a set-intersection query, which Mongo's aggregation
    pipeline handles natively via $setIntersection (no $graphLookup needed
    unless a future multi-hop traversal is added).

    Document shape (one per chunk, collection ``RAG_GRAPH_CHUNKS_COLLECTION``):
        {chunk_id, source, title, content, section, page_number,
         entities: [str], adjacent_to: [chunk_id]}
    """

    async def _collection(self) -> Any:
        from .mongo_connection import get_connection

        conn = await get_connection(uri=settings.mongo_uri, db_name=settings.mongo_db_name)
        return conn.get_collection(settings.graph_chunks_collection)

    async def upsert_chunks(self, chunks: list[Document]) -> int:
        rows = [_chunk_fields(c) for c in chunks]
        if not rows:
            return 0

        # Adjacency: link consecutive chunks that share a source document —
        # same forward-pass logic as Neo4jGraphStore's pairs computation.
        adjacency: dict[str, list[str]] = {}
        prev_by_source: dict[str, str] = {}
        for row in rows:
            src = row["source"]
            if src and src in prev_by_source:
                adjacency.setdefault(prev_by_source[src], []).append(row["chunk_id"])
            if src:
                prev_by_source[src] = row["chunk_id"]

        from pymongo import UpdateOne

        operations = [
            UpdateOne(
                {"chunk_id": row["chunk_id"]},
                {
                    "$set": {
                        "chunk_id": row["chunk_id"],
                        "source": row["source"],
                        "title": row["title"],
                        "content": row["content"],
                        "section": row["section"],
                        "page_number": row["page"],
                        "entities": row["entities"],
                        "adjacent_to": adjacency.get(row["chunk_id"], []),
                    }
                },
                upsert=True,
            )
            for row in rows
        ]

        collection = await self._collection()
        await collection.bulk_write(operations, ordered=False)
        logger.info(
            "MongoGraphStore: upserted %d chunk(s), %d adjacency edge(s).",
            len(rows),
            sum(len(v) for v in adjacency.values()),
        )
        return len(rows)

    async def retrieve(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ScoredDocument]:
        entities = _extract_entities(question)
        if not entities:
            return []  # nothing to traverse — let vector/keyword carry the query

        source = None
        if metadata_filter:
            source = metadata_filter.get("source") or metadata_filter.get("metadata.source")

        match: dict[str, Any] = {"entities": {"$in": entities}}
        if source:
            match["source"] = source

        pipeline = [
            {"$match": match},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "content": 1,
                    "source": 1,
                    "section": 1,
                    "page_number": 1,
                    "shared": {"$size": {"$setIntersection": ["$entities", entities]}},
                }
            },
            {"$sort": {"shared": -1}},
            {"$limit": int(top_k)},
        ]

        collection = await self._collection()
        records = [doc async for doc in collection.aggregate(pipeline)]

        denom = float(len(entities)) or 1.0
        out: list[ScoredDocument] = []
        for rec in records:
            shared = float(rec.get("shared", 0) or 0)
            doc = Document(
                page_content=rec.get("content") or "",
                metadata={
                    "source": rec.get("source") or "",
                    "section": rec.get("section") or "",
                    "page_number": rec.get("page_number"),
                    "chunk_id": rec.get("chunk_id") or "",
                },
            )
            out.append(ScoredDocument(document=doc, score=min(shared / denom, 1.0), source="graph"))
        logger.debug(
            "MongoGraphStore: retrieved %d graph candidate(s) for %d query entities.",
            len(out),
            len(entities),
        )
        return out

    async def delete_by_source(self, source: str) -> int:
        collection = await self._collection()
        result = await collection.delete_many({"source": source})
        return int(result.deleted_count)


# ─────────────────────────────────────────────────── Neo4j backend

class Neo4jGraphStore:
    """Relationship-aware store backed by Neo4j (async driver)."""

    def __init__(self, driver: Any, database: str | None = None) -> None:
        self._driver = driver
        self._database = database or None

    async def upsert_chunks(self, chunks: list[Document]) -> int:
        rows = [_chunk_fields(c) for c in chunks]
        if not rows:
            return 0

        # Adjacency: link consecutive chunks that share a source document.
        pairs: list[dict[str, str]] = []
        prev_by_source: dict[str, str] = {}
        for row in rows:
            src = row["source"]
            if src and src in prev_by_source:
                pairs.append({"a": prev_by_source[src], "b": row["chunk_id"]})
            if src:
                prev_by_source[src] = row["chunk_id"]

        async with self._driver.session(database=self._database) as session:
            await session.run(_UPSERT_CHUNKS_CYPHER, rows=rows)
            await session.run(_UPSERT_ENTITIES_CYPHER, rows=rows)
            if pairs:
                await session.run(_ADJ_CYPHER, pairs=pairs)

        logger.info(
            "Neo4jGraphStore: upserted %d chunk(s), %d adjacency edge(s).", len(rows), len(pairs)
        )
        return len(rows)

    async def retrieve(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ScoredDocument]:
        entities = _extract_entities(question)
        if not entities:
            return []  # nothing to traverse — let vector/keyword carry the query

        source = None
        if metadata_filter:
            source = metadata_filter.get("source") or metadata_filter.get("metadata.source")

        records: list[Any] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _RETRIEVE_CYPHER, entities=entities, top_k=int(top_k), source=source
            )
            async for rec in result:
                records.append(rec)

        denom = float(len(entities)) or 1.0
        out: list[ScoredDocument] = []
        for rec in records:
            shared = float(rec.get("shared", 0) or 0)
            doc = Document(
                page_content=rec.get("content") or "",
                metadata={
                    "source": rec.get("source") or "",
                    "section": rec.get("section") or "",
                    "page_number": rec.get("page"),
                    "chunk_id": rec.get("chunk_id") or "",
                },
            )
            out.append(
                ScoredDocument(document=doc, score=min(shared / denom, 1.0), source="graph")
            )
        logger.debug(
            "Neo4jGraphStore: retrieved %d graph candidate(s) for %d query entities.",
            len(out),
            len(entities),
        )
        return out

    async def delete_by_source(self, source: str) -> int:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(_DELETE_CYPHER, source=source)
            summary = await result.consume()
            return int(getattr(summary.counters, "nodes_deleted", 0))


# ─────────────────────────────────────────────────── backend resolution

_neo4j_driver_cache: dict[tuple[str, str], Any] = {}


def _build_neo4j_driver(uri: str, user: str, password: str) -> Any:
    """Create (and cache) an async Neo4j driver. Lazy-imports the driver."""
    key = (uri, user)
    cached = _neo4j_driver_cache.get(key)
    if cached is not None:
        return cached
    from neo4j import AsyncGraphDatabase  # lazy import — optional dependency

    auth = (user, password) if user else None
    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    _neo4j_driver_cache[key] = driver
    return driver


def _resolve_graph_backend() -> Any:
    """Pick the graph backend from config, degrading to a no-op on any problem."""
    if not settings.graph_retrieval_enabled:
        return NoopGraphStore(warn=False)

    backend = (settings.graph_db_backend or "neo4j").lower()
    if backend == "mongo":
        logger.info(
            "RAG graph store: MongoDB backend (collection=%s, same shared database)",
            settings.graph_chunks_collection,
        )
        return MongoGraphStore()

    if backend == "neo4j" and settings.graph_db_uri:
        try:
            driver = _build_neo4j_driver(
                settings.graph_db_uri, settings.graph_db_user, settings.graph_db_password
            )
            logger.info("RAG graph store: Neo4j backend at %s", settings.graph_db_uri)
            return Neo4jGraphStore(driver, database=settings.graph_db_database)
        except ImportError:
            logger.error(
                "RAG graph store: graph_retrieval_enabled but the 'neo4j' driver is not "
                "installed (pip install neo4j). Falling back to a no-op — only vector + "
                "keyword results are used."
            )
            return NoopGraphStore(warn=True)
        except Exception as exc:  # noqa: BLE001 — connection/config errors
            logger.error("RAG graph store: Neo4j init failed (%s) — falling back to no-op.", exc)
            return NoopGraphStore(warn=True)

    logger.warning(
        "RAG graph store: backend=%r uri=%r not usable (only 'neo4j' is implemented, "
        "and a URI is required) — graph retrieval is a no-op.",
        backend, settings.graph_db_uri,
    )
    return NoopGraphStore(warn=True)


class GraphStore:
    """Public wrapper — delegates to whichever backend config resolves to.

    ``ingestion.py`` and ``retrieval.py`` instantiate ``GraphStore()`` per
    node; the underlying Neo4j driver is cached module-side, so this stays
    cheap.
    """

    def __init__(self) -> None:
        self._impl = _resolve_graph_backend()

    async def upsert_chunks(self, chunks: list[Document]) -> int:
        return await self._impl.upsert_chunks(chunks)

    async def retrieve(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ScoredDocument]:
        return await self._impl.retrieve(question, top_k, metadata_filter)

    async def delete_by_source(self, source: str) -> int:
        return await self._impl.delete_by_source(source)

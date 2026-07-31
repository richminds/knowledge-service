"""MongoDB Atlas Vector Store for the Knowledge Service.

Storage schema (chunks collection):
  {
    "_id":      ObjectId,
    "chunk_id": str  (stable, derived from source+section+index),
    "content":  str,
    "embedding": [float, ...],   # length = RAG_EMBEDDING_DIMENSIONS (rag/config.py)
    "metadata": {
        "source":            str,
        "content_hash":      str,
        "authorized_users":  str | list[str],
        "authorized_teams":  str | list[str],
        ... (all LangChain Document metadata fields)
    }
  }

Atlas Vector Search index definition (create once per cluster):
  {
    "name": "<vector_index_name>",
    "type": "vectorSearch",
    "definition": {
      "fields": [
        { "type": "vector", "path": "embedding",
          "numDimensions": 1536, "similarity": "cosine" },
        { "type": "filter", "path": "metadata.authorized_users" },
        { "type": "filter", "path": "metadata.source" },
        { "type": "filter", "path": "metadata.content_hash" }
      ]
    }
  }

The index is created programmatically by ensure_indexes().  If the Atlas
tier does not allow programmatic index creation, create it via the UI using
the definition above.

Only the paths declared as "filter" type above can be used as $vectorSearch
pre-filters.  A metadata_filter on any other field makes the aggregation fail;
semantic_search() catches it and returns [], so the query silently loses its
semantic leg.  Add the field here and rebuild the index to filter on it.

Keyword search uses in-memory BM25 over a metadata-filtered document fetch
(up to RAG_KEYWORD_SEARCH_FETCH_LIMIT docs).  This avoids requiring a
separate Atlas Search index.  For very large corpora (>500k chunks) consider
upgrading to Atlas Full-Text Search ($search stage).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from .config import settings
from .llm_gateway_sdk import GatewayEmbeddingsClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────── shared type

@dataclass
class ScoredDocument:
    document: Document
    score: float
    source: str   # "semantic" | "keyword" | "hybrid" | "parent_child" | "reranked"


# ─────────────────────────────────────────────────── connection helper

async def _get_collection(name: str) -> Any:
    """Return the Motor collection, reusing the shared connection pool."""
    from .mongo_connection import get_connection

    conn = await get_connection(
        uri=settings.mongo_uri,
        db_name=settings.mongo_db_name,
    )
    return conn.get_collection(name)


# ──────────────────────────────────────────────── index management

_VECTOR_INDEX_DEFINITION = {
    "name": settings.vector_index_name,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": settings.embedding_dimensions,
                "similarity": settings.vector_similarity,
            },
            {"type": "filter", "path": "metadata.authorized_users"},
            {"type": "filter", "path": "metadata.source"},
            {"type": "filter", "path": "metadata.content_hash"},
        ]
    },
}

_REGULAR_INDEXES = [
    {"keys": [("chunk_id", 1)], "options": {"unique": True, "name": "chunk_id_unique"}},
    {"keys": [("metadata.content_hash", 1)], "options": {"name": "content_hash_idx"}},
    {"keys": [("metadata.source", 1)], "options": {"name": "source_idx"}},
    {"keys": [("content", "text")], "options": {"name": "content_text_idx"}},
]


async def ensure_indexes() -> None:
    """Create regular MongoDB indexes and attempt Atlas Vector Search index creation.

    Safe to call on every startup — all operations are idempotent.
    If Atlas rejects programmatic vector index creation (free-tier clusters
    require the UI), a warning is logged but startup continues.
    """
    from .mongo_connection import get_connection

    conn = await get_connection(uri=settings.mongo_uri, db_name=settings.mongo_db_name)
    col_name = settings.chunks_collection

    # Regular indexes (unique on chunk_id, hashed content_hash, text search)
    await conn.ensure_indexes(col_name, _REGULAR_INDEXES)

    # Atlas Vector Search index — requires M10+ cluster or Atlas Search enabled
    col = conn.get_collection(col_name)
    try:
        await col.database.command(
            "createSearchIndexes",
            col_name,
            indexes=[_VECTOR_INDEX_DEFINITION],
        )
        logger.info("Atlas Vector Search index '%s' created.", settings.vector_index_name)
    except Exception as exc:  # noqa: BLE001
        if "already exists" in str(exc).lower() or "IndexAlreadyExists" in type(exc).__name__:
            logger.debug("Atlas Vector Search index already exists — skipping creation.")
        else:
            logger.warning(
                "Could not create Atlas Vector Search index programmatically (%s). "
                "Create it manually in the Atlas UI:\n%s",
                exc,
                _VECTOR_INDEX_DEFINITION,
            )


# ──────────────────────────────────────────────────────────── VectorStore

class VectorStore:
    """Async MongoDB Atlas vector store.

    All public methods are coroutines and must be awaited.  The store reuses
    the shared connection pool so no extra client is opened.
    """

    def __init__(self, embeddings: GatewayEmbeddingsClient) -> None:
        self._embeddings = embeddings

    # ──────────────────────────────────────────────────────── ingestion

    async def upsert(self, chunks: list[Document]) -> int:
        """Embed and upsert chunk documents.

        Existing chunks with the same content_hash are deleted first so
        re-indexing a modified file produces clean results.

        Returns the number of documents inserted.
        """
        if not chunks:
            return 0

        col = await _get_collection(settings.chunks_collection)

        # Delete stale chunks for any content_hash in this batch
        hashes = sorted({
            str(c.metadata["content_hash"])
            for c in chunks
            if c.metadata.get("content_hash")
        })
        if hashes:
            await self.delete_by_content_hashes(hashes)

        # Embed all chunk texts in one batched call, off the event loop's
        # synchronous path — this is a real network round trip to the LLM
        # Gateway now, not an in-process library call.
        texts = [c.page_content for c in chunks]
        embeddings = await self._embeddings.aembed_documents(texts)

        docs = [
            {
                "chunk_id": str(chunk.metadata["chunk_id"]),
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": _serialise_metadata(chunk.metadata),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        # Bulk upsert — replace_one with upsert=True per chunk_id
        from pymongo import ReplaceOne

        ops = [
            ReplaceOne({"chunk_id": doc["chunk_id"]}, doc, upsert=True)
            for doc in docs
        ]
        result = await col.bulk_write(ops, ordered=False)
        inserted = result.upserted_count + result.modified_count
        logger.info("VectorStore.upsert: %d chunk(s) written.", inserted)
        return inserted

    async def delete_by_content_hashes(self, content_hashes: list[str]) -> int:
        """Delete all chunks whose content_hash is in the provided list."""
        if not content_hashes:
            return 0
        col = await _get_collection(settings.chunks_collection)
        result = await col.delete_many({"metadata.content_hash": {"$in": content_hashes}})
        logger.debug(
            "Deleted %d chunk(s) for %d content hash(es).",
            result.deleted_count,
            len(content_hashes),
        )
        return result.deleted_count

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks originating from a given source file path."""
        col = await _get_collection(settings.chunks_collection)
        result = await col.delete_many({"metadata.source": source})
        logger.info("Deleted %d chunk(s) for source '%s'.", result.deleted_count, source)
        return result.deleted_count

    # ──────────────────────────────────────────────────────── retrieval

    async def semantic_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """Run $vectorSearch against the Atlas vector index.

        metadata_filter values are applied as Atlas pre-filter fields.
        Only fields declared as "filter" type in the index are supported.
        """
        col = await _get_collection(settings.chunks_collection)
        query_embedding = await self._embeddings.aembed_query(query)

        vector_search_stage: dict[str, Any] = {
            "index": settings.vector_index_name,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": max(settings.retrieval_num_candidates, top_k * 10),
            "limit": top_k,
        }
        if metadata_filter:
            vector_search_stage["filter"] = _build_atlas_filter(metadata_filter)

        pipeline = [
            {"$vectorSearch": vector_search_stage},
            {
                "$project": {
                    "chunk_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            cursor = col.aggregate(pipeline)
            results: list[ScoredDocument] = []
            async for doc in cursor:
                results.append(
                    ScoredDocument(
                        document=Document(
                            page_content=doc["content"],
                            metadata=doc.get("metadata", {}),
                        ),
                        score=float(doc.get("score", 0.0)),
                        source="semantic",
                    )
                )
            return results
        except Exception as exc:
            logger.error(
                "Atlas $vectorSearch failed — index may not exist yet. "
                "Run ensure_indexes() on startup. Error: %s", exc
            )
            return []

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """BM25 keyword search over a metadata-filtered document subset.

        Fetches up to RAG_KEYWORD_SEARCH_FETCH_LIMIT documents matching the
        filter, builds a BM25 index in memory, and returns the top-k by score.
        """
        col = await _get_collection(settings.chunks_collection)

        mongo_filter = _build_mongo_filter(metadata_filter) if metadata_filter else {}
        cursor = col.find(mongo_filter, {"chunk_id": 1, "content": 1, "metadata": 1})
        raw_docs = await cursor.limit(settings.keyword_search_fetch_limit).to_list(
            settings.keyword_search_fetch_limit
        )

        if not raw_docs:
            return []

        from rank_bm25 import BM25Okapi  # lazy — optional dependency

        texts = [d["content"] for d in raw_docs]
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(_tokenize(query))

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            ScoredDocument(
                document=Document(
                    page_content=raw_docs[i]["content"],
                    metadata=raw_docs[i].get("metadata", {}),
                ),
                score=float(scores[i]),
                source="keyword",
            )
            for i in ranked_indices
            if scores[i] > 0
        ]

    async def count(self, metadata_filter: dict[str, Any] | None = None) -> int:
        """Return the number of stored chunks matching an optional filter."""
        col = await _get_collection(settings.chunks_collection)
        mongo_filter = _build_mongo_filter(metadata_filter) if metadata_filter else {}
        return await col.count_documents(mongo_filter)


# ──────────────────────────────────────────────── parent-child expansion

async def expand_parent_context(candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Replace child chunk content with its broader parent context.

    Deduplicates by parent_id so only one expanded document is returned
    per parent group, avoiding repeated context in the final prompt.
    """
    from .parent_store import MongoParentStore

    store = MongoParentStore()
    expanded: list[ScoredDocument] = []
    seen_parent_ids: set[str] = set()

    for candidate in candidates:
        metadata = dict(candidate.document.metadata)
        parent_id = str(metadata.get("parent_id") or metadata.get("chunk_id"))

        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        parent_doc = await store.load(parent_id)
        if parent_doc:
            parent_content = parent_doc.page_content
            metadata.update(parent_doc.metadata)
        else:
            parent_content = candidate.document.page_content

        metadata["expanded_from_child_id"] = metadata.get("chunk_id")
        expanded.append(
            ScoredDocument(
                document=Document(page_content=parent_content, metadata=metadata),
                score=candidate.score,
                source="parent_child",
            )
        )

    return expanded


# ──────────────────────────────────────────────────────── hybrid fusion

def hybrid_fusion(
    semantic_results: list[ScoredDocument],
    keyword_results: list[ScoredDocument],
    top_k: int,
    rrf_k: int = 60,
    graph_results: list[ScoredDocument] | None = None,
    web_results: list[ScoredDocument] | None = None,
) -> list[ScoredDocument]:
    """Reciprocal Rank Fusion across semantic, keyword, graph, and web results.

    Each result set contributes 1/(rrf_k + rank) to a document's fused score.
    A chunk appearing near the top of multiple lists receives a strong combined
    score.  Duplicates are collapsed by chunk identity.
    """
    by_id: dict[str, ScoredDocument] = {}
    fused_scores: dict[str, float] = {}

    for result_set in [semantic_results, keyword_results, graph_results or [], web_results or []]:
        for rank, result in enumerate(result_set, start=1):
            doc_id = _doc_id(result.document)
            by_id[doc_id] = result
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        ScoredDocument(
            document=by_id[doc_id].document,
            score=score,
            source="hybrid",
        )
        for doc_id, score in ranked
    ]


# ─────────────────────────────────────────────────────── private helpers

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _doc_id(document: Document) -> str:
    return str(
        document.metadata.get("chunk_id")
        or document.metadata.get("source")
        or hash(document.page_content)
    )


def _serialise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert metadata values to MongoDB-safe types (no Path, no complex objects)."""
    return {k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
            for k, v in metadata.items()}


def _build_atlas_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat metadata_filter dict to an Atlas pre-filter expression.

    Atlas pre-filter fields must be declared as "filter" type in the index.
    Keys are automatically prefixed with "metadata." if not already nested.
    """
    conditions = {}
    for k, v in metadata_filter.items():
        field = k if k.startswith("metadata.") else f"metadata.{k}"
        conditions[field] = v
    return conditions


def _build_mongo_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """Convert a metadata_filter dict to a regular MongoDB query filter."""
    return {
        (k if k.startswith("metadata.") else f"metadata.{k}"): v
        for k, v in metadata_filter.items()
    }

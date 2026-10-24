"""Vector store on MongoDB Atlas: one collection of chunk documents (`_id` = chunk_id).

Retrieval modes:
  vector  - Atlas Vector Search ($vectorSearch, cosine, HNSW). Atlas reports cosine as
            (1 + cos) / 2; it is mapped back to cos so scores match RETRIEVAL_MIN_SCORE etc.
  keyword - Atlas Search ($search, Lucene BM25) over title + section_path + content
  hybrid  - both lists fused with Reciprocal Rank Fusion (k=60)
  semantic_hybrid - hybrid, then the fused candidates are reranked in the app
            (MONGODB_SEMANTIC_RERANKER: the utility LLM or a Cohere endpoint)

Every read applies collection, ACL (C-192) and metadata filters inside the query: as a
$vectorSearch pre-filter, as $search compound filters, or as a find() filter. The two Atlas
Search indexes are created by ensure_index(); they build asynchronously, so a brand-new
collection answers searches a few seconds after the first write.
"""

import asyncio
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.errors import CollectionInvalid, OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.core.mongo import get_mongo_database
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import (
    SEARCHABLE_KINDS,
    VectorRecord,
    VectorSearchResult,
    VectorStore,
)

logger = get_logger(__name__)

NO_PRINCIPAL = "\x00no-principal"
_RRF_K = 60
_MAX_NUM_CANDIDATES = 10_000  # Atlas limit for $vectorSearch numCandidates
_UPSERT_BATCH = 500
_TEXT_PATHS = ["title", "section_path", "content"]
# Fields every read may filter on: indexed as `filter` (vector index) / `token` (search index).
_FILTER_FIELDS = ("collection", "kind", "acl", "doc_type", "language", "source", "tags", "doc_id")
_RESULT_FIELDS = (
    "content", "source", "metadata", "doc_id", "title", "section_path", "acl",
    "effective_date", "page", "chunk_index", "parent_id", "prev_id", "next_id", "kind",
    "target_id",
)  # fmt: skip
_PROJECTION = {field: 1 for field in _RESULT_FIELDS}


def parse_date(value: str | None) -> datetime | None:
    """ISO date or datetime -> aware datetime (UTC when no offset is given)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _principals(filters: RetrievalFilters) -> list[str]:
    """The caller's principals for an ACL match. Atlas rejects an empty `in` list, so a caller
    with no principals gets one that no ACL contains: it sees nothing (fail closed)."""
    return sorted(filters.principals or ()) or [NO_PRINCIPAL]


def _date_range(filters: RetrievalFilters) -> dict[str, datetime]:
    bounds: dict[str, datetime] = {}
    start = parse_date(filters.effective_from)
    if start:
        bounds["$gte"] = start
    end = parse_date(filters.effective_to)
    if end:
        bounds["$lt"] = end + timedelta(days=1)  # inclusive end date
    return bounds


def build_filter(filters: RetrievalFilters, kinds: tuple[str, ...] | None = None) -> dict[str, Any]:
    """MQL filter for collection, ACL, kind and metadata. Valid both in find() and as a
    $vectorSearch pre-filter (only $eq / $in / range operators, combined with $and)."""
    clauses: list[dict[str, Any]] = [{"collection": filters.collection}]
    if filters.principals is not None:
        clauses.append({"acl": {"$in": _principals(filters)}})
    if kinds is not None:
        clauses.append({"kind": {"$in": list(kinds)}})
    for field, values in (
        ("doc_type", filters.doc_types),
        ("language", filters.languages),
        ("source", filters.sources),
        ("tags", filters.tags),
        ("doc_id", filters.doc_ids),
    ):
        if values:
            clauses.append({field: {"$in": sorted(values)}})
    dates = _date_range(filters)
    if dates:
        clauses.append({"effective_date": dates})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def build_search_filters(
    filters: RetrievalFilters, kinds: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """The same restrictions as Atlas Search compound `filter` clauses (no score impact)."""
    clauses: list[dict[str, Any]] = [
        {"equals": {"path": "collection", "value": filters.collection}}
    ]
    if filters.principals is not None:
        clauses.append({"in": {"path": "acl", "value": _principals(filters)}})
    if kinds is not None:
        clauses.append({"in": {"path": "kind", "value": list(kinds)}})
    for field, values in (
        ("doc_type", filters.doc_types),
        ("language", filters.languages),
        ("source", filters.sources),
        ("tags", filters.tags),
        ("doc_id", filters.doc_ids),
    ):
        if values:
            clauses.append({"in": {"path": field, "value": sorted(values)}})
    dates = _date_range(filters)
    if dates:
        bounds = {op.lstrip("$"): value for op, value in dates.items()}
        clauses.append({"range": {"path": "effective_date", **bounds}})
    return clauses


def vector_index_definition(dimensions: int) -> dict[str, Any]:
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dimensions,
             "similarity": "cosine"},
            *({"type": "filter", "path": field} for field in _FILTER_FIELDS),
            {"type": "filter", "path": "effective_date"},
        ]
    }  # fmt: skip


def search_index_definition(analyzer: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        path: {"type": "string", "analyzer": analyzer} for path in _TEXT_PATHS
    }
    fields.update({field: {"type": "token"} for field in _FILTER_FIELDS})
    fields["effective_date"] = {"type": "date"}
    return {"mappings": {"dynamic": False, "fields": fields}}


def rrf_fuse(rankings: list[list[VectorSearchResult]], top_k: int) -> list[VectorSearchResult]:
    """Reciprocal Rank Fusion: results ranked high by several retrievers float to the top."""
    scores: dict[str, float] = {}
    first_seen: dict[str, VectorSearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            first_seen.setdefault(result.chunk_id, result)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    for chunk_id in ordered:
        first_seen[chunk_id].score = round(scores[chunk_id], 6)
    return [first_seen[chunk_id] for chunk_id in ordered]


def _to_document(record: VectorRecord) -> dict[str, Any]:
    document = {
        "content": record.content,
        "title": record.title,
        "section_path": record.section_path,
        "source": record.source,
        "collection": record.collection,
        "doc_id": record.doc_id,
        "kind": record.kind,
        "target_id": record.target_id,
        "parent_id": record.parent_id,
        "prev_id": record.prev_id,
        "next_id": record.next_id,
        "chunk_index": record.chunk_index,
        "page": record.page,
        "acl": list(record.acl),
        "doc_type": record.doc_type,
        "language": record.language,
        "tags": list(record.tags),
        "effective_date": parse_date(record.effective_date),
        "content_hash": record.content_hash,
        "embedding_model": record.embedding_model,
        "ingest_version": record.ingest_version,
        "metadata": dict(record.metadata),
    }
    # Records without a vector (e.g. parents) keep the one already stored.
    if record.embedding is not None:
        document["embedding"] = [float(x) for x in record.embedding]
    return document


def _to_result(doc: dict[str, Any], score: float) -> VectorSearchResult:
    effective = doc.get("effective_date")
    return VectorSearchResult(
        chunk_id=str(doc["_id"]),
        content=doc.get("content") or "",
        score=score,
        source=doc.get("source") or "",
        metadata={str(k): str(v) for k, v in (doc.get("metadata") or {}).items()},
        doc_id=doc.get("doc_id") or "",
        title=doc.get("title") or "",
        section_path=doc.get("section_path") or "",
        acl=list(doc.get("acl") or []),
        effective_date=effective.isoformat() if isinstance(effective, datetime) else effective,
        page=doc.get("page"),
        chunk_index=doc.get("chunk_index") or 0,
        parent_id=doc.get("parent_id") or "",
        prev_id=doc.get("prev_id") or "",
        next_id=doc.get("next_id") or "",
        kind=doc.get("kind") or "chunk",
        target_id=doc.get("target_id") or "",
    )


class MongoVectorStore(VectorStore):
    """Vector store backed by MongoDB Atlas Vector Search + Atlas Search."""

    def __init__(
        self,
        *,
        url: str,
        database: str,
        collection: str = "rag_chunks",
        vector_index: str = "rag_chunks_vector",
        search_index: str = "rag_chunks_text",
        embedding_dim: int = 3072,
        retrieval_mode: str = "hybrid",
        text_analyzer: str = "lucene.english",
        num_candidates_factor: int = 10,
        semantic_reranker=None,
        semantic_candidates: int = 20,
    ) -> None:
        if not url:
            raise VectorStoreError("MONGODB_URL is not configured")
        self._url = url
        self._database_name = database
        self._collection_name = collection
        self._vector_index = vector_index
        self._search_index = search_index
        self._dim = int(embedding_dim)
        self._retrieval_mode = retrieval_mode
        self._analyzer = text_analyzer
        self._num_candidates_factor = max(1, int(num_candidates_factor))
        # semantic_hybrid: rerank the fused candidates in the app.
        self._semantic_reranker = semantic_reranker if retrieval_mode == "semantic_hybrid" else None
        self._semantic_candidates = int(semantic_candidates)

    @property
    def _db(self):
        return get_mongo_database(self._url, self._database_name)

    @property
    def _collection(self):
        return self._db[self._collection_name]

    # -- lifecycle ------------------------------------------------------------------------
    async def ensure_index(self) -> None:
        """Create the collection, regular indexes and both Atlas Search indexes (idempotent)."""
        try:
            try:
                await self._db.create_collection(self._collection_name)
            except CollectionInvalid:
                pass  # already exists
            collection = self._collection
            await collection.create_index([("collection", ASCENDING), ("kind", ASCENDING)])
            await collection.create_index([("collection", ASCENDING), ("source", ASCENDING)])
            await collection.create_index([("doc_id", ASCENDING), ("chunk_index", ASCENDING)])
            existing = {
                index["name"]: index async for index in await collection.list_search_indexes()
            }
            wanted = [
                SearchIndexModel(
                    definition=vector_index_definition(self._dim),
                    name=self._vector_index,
                    type="vectorSearch",
                ),
                SearchIndexModel(
                    definition=search_index_definition(self._analyzer),
                    name=self._search_index,
                    type="search",
                ),
            ]
            for model in wanted:
                name = model.document["name"]
                if name in existing:
                    continue
                await collection.create_search_index(model)
                logger.info("mongo_search_index_created", index=name)
        except OperationFailure as exc:
            raise VectorStoreError(f"MongoDB index setup failed: {exc}") from exc
        logger.info("mongo_vector_store_ready", collection=self._collection_name, dims=self._dim)

    async def ping(self) -> None:
        try:
            await self._db.command("ping")
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB is unreachable: {exc}") from exc

    # -- writes ---------------------------------------------------------------------------
    async def upsert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        operations = [
            UpdateOne({"_id": r.chunk_id}, {"$set": _to_document(r)}, upsert=True) for r in records
        ]
        try:
            for start in range(0, len(operations), _UPSERT_BATCH):
                await self._collection.bulk_write(
                    operations[start : start + _UPSERT_BATCH], ordered=False
                )
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB upsert failed: {exc}") from exc
        return len(operations)

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        try:
            result = await self._collection.update_many(
                {"collection": collection, "source": source}, {"$set": {"acl": list(acl)}}
            )
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB ACL update failed: {exc}") from exc
        return result.matched_count

    async def delete_by_source(
        self,
        source: str,
        collection: str,
        except_version: str | None = None,
        keep_ids: Collection[str] = (),
    ) -> int:
        query: dict[str, Any] = {"collection": collection, "source": source}
        if except_version:
            query["ingest_version"] = {"$ne": except_version}
        if keep_ids:
            query["_id"] = {"$nin": list(keep_ids)}
        return await self._delete(query)

    async def delete_collection(self, collection: str) -> int:
        return await self._delete({"collection": collection})

    async def _delete(self, query: dict[str, Any]) -> int:
        try:
            result = await self._collection.delete_many(query)
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB delete failed: {exc}") from exc
        return result.deleted_count

    # -- reads ----------------------------------------------------------------------------
    async def _aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            cursor = await self._collection.aggregate(pipeline)
            return await cursor.to_list()
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB search failed: {exc}") from exc

    async def _vector_search(
        self, embedding: list[float], filters: RetrievalFilters, kinds, limit: int
    ) -> list[VectorSearchResult]:
        num_candidates = min(max(limit * self._num_candidates_factor, 100), _MAX_NUM_CANDIDATES)
        docs = await self._aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": self._vector_index,
                        "path": "embedding",
                        "queryVector": [float(x) for x in embedding],
                        "numCandidates": num_candidates,
                        "limit": min(limit, num_candidates),
                        "filter": build_filter(filters, kinds),
                    }
                },
                {"$project": {**_PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
            ]
        )
        return [_to_result(doc, 2.0 * float(doc.get("score", 0.5)) - 1.0) for doc in docs]

    async def _keyword_search(
        self, text: str, filters: RetrievalFilters, kinds, limit: int
    ) -> list[VectorSearchResult]:
        if not text.strip():
            return []
        docs = await self._aggregate(
            [
                {
                    "$search": {
                        "index": self._search_index,
                        "compound": {
                            "must": [{"text": {"query": text, "path": _TEXT_PATHS}}],
                            "filter": build_search_filters(filters, kinds),
                        },
                    }
                },
                {"$limit": limit},
                {"$project": {**_PROJECTION, "score": {"$meta": "searchScore"}}},
            ]
        )
        return [_to_result(doc, float(doc.get("score", 0.0))) for doc in docs]

    def _keyword_with_vectors(self, hybrid: bool | None) -> bool:
        """Whether a query with an embedding also runs the keyword side (per-call override)."""
        if hybrid is not None:
            return hybrid
        return self._retrieval_mode in ("hybrid", "semantic_hybrid")

    async def search(
        self,
        query_embedding: list[float] | None,
        filters: RetrievalFilters,
        top_k: int,
        query_text: str | None = None,
        kinds: tuple[str, ...] = SEARCHABLE_KINDS,
        hybrid: bool | None = None,
    ) -> list[VectorSearchResult]:
        use_keyword = bool(query_text) and (
            query_embedding is None or self._keyword_with_vectors(hybrid)
        )
        if query_embedding is not None and not use_keyword:
            return await self._vector_search(query_embedding, filters, kinds, top_k)
        if query_embedding is None:
            return await self._keyword_search(query_text or "", filters, kinds, top_k)
        pool = max(top_k * 2, 20)
        vector, keyword = await asyncio.gather(
            self._vector_search(query_embedding, filters, kinds, pool),
            self._keyword_search(query_text or "", filters, kinds, pool),
        )
        if self._semantic_reranker is None:
            return rrf_fuse([vector, keyword], top_k)
        candidates = rrf_fuse([vector, keyword], max(top_k, self._semantic_candidates))
        return await self._semantic_reranker.rerank(query_text or "", candidates, top_k)

    async def _find(
        self, query: dict[str, Any], sort: list | None = None, limit: int = 0
    ) -> list[VectorSearchResult]:
        try:
            cursor = self._collection.find(query, _PROJECTION, sort=sort, limit=limit)
            return [_to_result(doc, 0.0) async for doc in cursor]
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB query failed: {exc}") from exc

    async def get_by_ids(
        self, ids: list[str], filters: RetrievalFilters
    ) -> list[VectorSearchResult]:
        if not ids:
            return []
        query = {"$and": [build_filter(filters, kinds=None), {"_id": {"$in": list(ids)}}]}
        found = {r.chunk_id: r for r in await self._find(query)}
        return [found[i] for i in ids if i in found]

    async def get_document_chunks(
        self, doc_id: str, filters: RetrievalFilters, limit: int = 200
    ) -> list[VectorSearchResult]:
        query = {"$and": [build_filter(filters, kinds=("chunk",)), {"doc_id": doc_id}]}
        return await self._find(query, sort=[("chunk_index", ASCENDING)], limit=limit)

    async def count(self, collection: str, source: str | None = None) -> int:
        query: dict[str, Any] = {"collection": collection}
        if source:
            query["source"] = source
        try:
            return await self._collection.count_documents(query)
        except PyMongoError as exc:
            raise VectorStoreError(f"MongoDB count failed: {exc}") from exc

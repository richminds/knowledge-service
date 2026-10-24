"""Graph store on MongoDB Atlas: the graph model of base.py as five collections.

  <prefix>_chunks     _id = chunk_id; entity_keys = the MENTIONS edges, prev_id = NEXT,
                      section_key / doc_id = PART_OF
  <prefix>_sections   _id = "<collection>|<doc_id>|<section>"
  <prefix>_documents  _id = "<collection>|<doc_id>"
  <prefix>_entities   unique (collection, key); rank, community_<level>, and the
                      description embedding for entity linking (G-25, G-37)
  <prefix>_relations  RELATED edges: src/dst entity keys, one per (type, chunk) (G-11)
  <prefix>_claims     Claim -ABOUT-> entity_key (G-12)

Traversal is done hop by hop with indexed queries (no server-side graph language), so
Cypher (G-43..G-45) is not supported: GRAPH_SEARCH_MODE=cypher falls back to local search.

ACL (G-51) is enforced in the queries: an entity is visible when a chunk the caller may
read mentions it; relations, claims and chunks carry their chunk's ACL.
"""

import random
import re
from difflib import SequenceMatcher
from typing import Any

from pymongo import ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import CollectionInvalid, OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

from app.core.exceptions import GraphStoreError
from app.core.logging import get_logger
from app.core.mongo import get_mongo_database
from app.rag.graph.base import (
    ChunkGraph,
    Claim,
    Entity,
    GraphExport,
    GraphFact,
    GraphStore,
    entity_key,
)
from app.rag.types import RetrievalFilters, is_allowed
from app.rag.vectorstore.base import VectorSearchResult

logger = get_logger(__name__)

_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_PROPERTY = re.compile(r"^(rank|community_\d{1,2})$")
_NO_PRINCIPAL = "\x00no-principal"
_ENTITY_FIELDS = {
    "key": 1, "name": 1, "type": 1, "description": 1, "rank": 1, "community_0": 1,
    "external_id": 1,
}  # fmt: skip
_COUNTED_LABELS = ("Entity", "Chunk", "Document", "Section", "Claim")


def _entity(doc: dict[str, Any]) -> Entity:
    community = doc.get("community_0")
    return Entity(
        name=doc.get("name") or doc.get("key") or "",
        type=doc.get("type") or "Concept",
        description=doc.get("description") or "",
        rank=float(doc.get("rank") or 0.0),
        community=community if isinstance(community, int) else None,
        external_id=doc.get("external_id") or "",
    )


def _acl_clause(filters: RetrievalFilters) -> dict[str, Any]:
    """ACL and account restriction for chunks / relations / claims (empty when neither is
    enforced)."""
    clause: dict[str, Any] = {}
    if filters.principals is not None:
        clause["acl"] = {"$in": sorted(filters.principals) or [_NO_PRINCIPAL]}
    if filters.account_id is not None:
        clause["account_id"] = filters.account_id
    return clause


class MongoGraphStore(GraphStore):
    def __init__(
        self,
        *,
        url: str,
        database: str,
        prefix: str = "rag_graph",
        entity_vector_index: str = "rag_graph_entities_vector",
        embedding_dim: int = 3072,
    ) -> None:
        if not url:
            raise GraphStoreError("MONGODB_URL is not configured")
        if not _PREFIX.match(prefix):
            raise GraphStoreError(f"Invalid graph collection prefix: {prefix!r}")
        self._url = url
        self._database_name = database
        self._prefix = prefix
        self._entity_vector_index = entity_vector_index
        self._embedding_dim = int(embedding_dim)

    # -- plumbing -------------------------------------------------------------------------
    @property
    def _db(self):
        return get_mongo_database(self._url, self._database_name)

    def _col(self, name: str):
        return self._db[f"{self._prefix}_{name}"]

    @property
    def _chunks(self):
        return self._col("chunks")

    @property
    def _entities(self):
        return self._col("entities")

    @property
    def _relations(self):
        return self._col("relations")

    @property
    def _claims(self):
        return self._col("claims")

    async def _visible_keys(
        self, collection: str, keys: list[str], filters: RetrievalFilters
    ) -> set[str]:
        """Entity keys mentioned by at least one chunk the caller may read."""
        if not keys:
            return set()
        query = {"collection": collection, "entity_keys": {"$in": keys}, **_acl_clause(filters)}
        mentioned = await self._chunks.distinct("entity_keys", query)
        return set(keys) & set(mentioned)

    async def _entities_by_key(
        self, collection: str, keys: list[str] | set[str]
    ) -> dict[str, dict[str, Any]]:
        if not keys:
            return {}
        cursor = self._entities.find(
            {"collection": collection, "key": {"$in": list(keys)}}, _ENTITY_FIELDS
        )
        return {doc["key"]: doc async for doc in cursor}

    async def _names(self, collection: str, keys: set[str]) -> dict[str, str]:
        return {
            key: doc.get("name") or key
            for key, doc in (await self._entities_by_key(collection, keys)).items()
        }

    # -- lifecycle ------------------------------------------------------------------------
    async def ensure_schema(self) -> None:
        try:
            for name in ("chunks", "sections", "documents", "entities", "relations", "claims"):
                try:
                    await self._db.create_collection(f"{self._prefix}_{name}")
                except CollectionInvalid:
                    pass
            await self._chunks.create_index([("collection", ASCENDING), ("source", ASCENDING)])
            await self._chunks.create_index([("collection", ASCENDING), ("entity_keys", ASCENDING)])
            await self._entities.create_index(
                [("collection", ASCENDING), ("key", ASCENDING)], unique=True
            )
            await self._entities.create_index(
                [("collection", ASCENDING), ("alias_keys", ASCENDING)]
            )
            await self._relations.create_index(
                [("collection", ASCENDING), ("src", ASCENDING), ("dst", ASCENDING),
                 ("type", ASCENDING), ("chunk_id", ASCENDING)],
                unique=True,
            )  # fmt: skip
            await self._relations.create_index([("collection", ASCENDING), ("dst", ASCENDING)])
            await self._relations.create_index([("collection", ASCENDING), ("source", ASCENDING)])
            await self._claims.create_index([("collection", ASCENDING), ("entity_key", ASCENDING)])
            await self._claims.create_index([("collection", ASCENDING), ("source", ASCENDING)])
            for name in ("sections", "documents"):
                await self._col(name).create_index(
                    [("collection", ASCENDING), ("source", ASCENDING)]
                )
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph schema setup failed: {exc}") from exc
        await self._ensure_entity_vector_index()
        logger.info("graph_schema_ready", prefix=self._prefix)

    async def _ensure_entity_vector_index(self) -> None:
        """Atlas Vector Search index for entity linking (G-25). Skipped when unavailable."""
        try:
            existing = {index["name"] async for index in await self._entities.list_search_indexes()}
            if self._entity_vector_index in existing:
                return
            await self._entities.create_search_index(
                SearchIndexModel(
                    definition={
                        "fields": [
                            {"type": "vector", "path": "embedding",
                             "numDimensions": self._embedding_dim, "similarity": "cosine"},
                            {"type": "filter", "path": "collection"},
                        ]
                    },
                    name=self._entity_vector_index,
                    type="vectorSearch",
                )
            )  # fmt: skip
        except OperationFailure as exc:
            logger.warning("entity_embeddings_unavailable", error=str(exc))

    async def ping(self) -> None:
        try:
            await self._db.command("ping")
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph DB is unreachable: {exc}") from exc

    # -- writes ---------------------------------------------------------------------------
    async def upsert_chunk_graph(self, chunk: ChunkGraph) -> None:
        collection = chunk.collection
        section = " > ".join(chunk.section_path) or chunk.title
        section_key = f"{chunk.doc_id}|{section}"
        entity_keys = sorted({e.key for e in chunk.entities})
        try:
            await self._chunks.update_one(
                {"_id": chunk.chunk_id},
                {
                    "$set": {
                        "collection": collection,
                        "source": chunk.source,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "acl": list(chunk.acl),
                        "account_id": chunk.account_id,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "section_key": section_key,
                        "prev_id": chunk.prev_id,  # G-20 reading order (NEXT)
                    },
                    "$addToSet": {"entity_keys": {"$each": entity_keys}},
                },
                upsert=True,
            )
            await self._col("documents").update_one(
                {"_id": f"{collection}|{chunk.doc_id}"},
                {"$set": {"collection": collection, "doc_id": chunk.doc_id,
                          "title": chunk.title, "source": chunk.source}},
                upsert=True,
            )  # fmt: skip
            await self._col("sections").update_one(
                {"_id": f"{collection}|{section_key}"},
                {"$set": {"collection": collection, "key": section_key, "name": section,
                          "doc_id": chunk.doc_id, "source": chunk.source}},
                upsert=True,
            )  # fmt: skip
            if chunk.entities:
                await self._entities.bulk_write(
                    [
                        UpdateOne(
                            {"collection": collection, "key": entity.key},
                            {
                                "$set": {"name": entity.name, "type": entity.type},
                                # First writer wins, like coalesce() on the existing value.
                                "$setOnInsert": {
                                    "description": entity.description,
                                    "alias_keys": sorted({entity_key(a) for a in entity.aliases}),
                                    "external_id": entity.external_id,
                                },
                            },
                            upsert=True,
                        )
                        for entity in chunk.entities
                    ],
                    ordered=False,
                )
            await self._write_relations(chunk, set(entity_keys))
            await self._write_claims(chunk, set(entity_keys))
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph write failed: {exc}") from exc

    async def _existing_keys(self, collection: str, keys: set[str], known: set[str]) -> set[str]:
        missing = keys - known
        if not missing:
            return keys
        found = await self._entities.distinct(
            "key", {"collection": collection, "key": {"$in": sorted(missing)}}
        )
        return (keys & known) | set(found)

    async def _write_relations(self, chunk: ChunkGraph, known: set[str]) -> None:
        if not chunk.relations:
            return
        pairs = [(entity_key(r.source), entity_key(r.target), r) for r in chunk.relations]
        # Like MATCH (a), (b) MERGE (a)-[r]->(b): only between entities that exist.
        existing = await self._existing_keys(
            chunk.collection, {k for src, dst, _ in pairs for k in (src, dst)}, known
        )
        operations = [
            UpdateOne(
                {"collection": chunk.collection, "src": src, "dst": dst, "type": relation.type,
                 "chunk_id": chunk.chunk_id},
                {"$set": {"description": relation.description, "source": chunk.source,
                          "acl": list(chunk.acl), "account_id": chunk.account_id,
                          "weight": float(relation.weight)}},
                upsert=True,
            )
            for src, dst, relation in pairs
            if src in existing and dst in existing
        ]  # fmt: skip
        if operations:
            await self._relations.bulk_write(operations, ordered=False)

    async def _write_claims(self, chunk: ChunkGraph, known: set[str]) -> None:
        if not chunk.claims:
            return
        subjects = {entity_key(c.subject) for c in chunk.claims}
        existing = await self._existing_keys(chunk.collection, subjects, known)
        documents = [
            {
                "collection": chunk.collection,
                "entity_key": entity_key(claim.subject),
                "text": claim.text,
                "date": claim.date or "",
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "acl": list(chunk.acl),
                "account_id": chunk.account_id,
            }
            for claim in chunk.claims
            if entity_key(claim.subject) in existing
        ]
        if documents:
            await self._claims.insert_many(documents, ordered=False)

    async def delete_by_source(self, source: str, collection: str) -> int:
        scope = {"collection": collection, "source": source}
        try:
            await self._relations.delete_many(scope)
            await self._claims.delete_many(scope)
            deleted = (await self._chunks.delete_many(scope)).deleted_count
            await self._col("sections").delete_many(scope)
            await self._col("documents").delete_many(scope)
            await self._delete_orphan_entities(collection)
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph delete failed: {exc}") from exc
        return deleted

    async def delete_collection(self, collection: str) -> int:
        deleted = 0
        try:
            for name in ("chunks", "sections", "documents", "entities", "claims"):
                result = await self._col(name).delete_many({"collection": collection})
                deleted += result.deleted_count
            await self._relations.delete_many({"collection": collection})
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph delete failed: {exc}") from exc
        return deleted

    async def _delete_orphan_entities(self, collection: str) -> None:
        """G-32: entities no chunk mentions any more are removed with their edges."""
        mentioned = await self._chunks.distinct("entity_keys", {"collection": collection})
        orphans = await self._entities.distinct(
            "key", {"collection": collection, "key": {"$nin": mentioned}}
        )
        if not orphans:
            return
        await self._entities.delete_many({"collection": collection, "key": {"$in": orphans}})
        await self._relations.delete_many(
            {
                "collection": collection,
                "$or": [{"src": {"$in": orphans}}, {"dst": {"$in": orphans}}],
            }
        )
        await self._claims.delete_many({"collection": collection, "entity_key": {"$in": orphans}})

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        scope = {"collection": collection, "source": source}
        change = {"$set": {"acl": list(acl)}}
        try:
            result = await self._chunks.update_many(scope, change)
            await self._relations.update_many(scope, change)
            await self._claims.update_many(scope, change)
        except PyMongoError as exc:
            raise GraphStoreError(f"Graph ACL update failed: {exc}") from exc
        return result.matched_count

    async def upsert_entity_embeddings(
        self, collection: str, items: list[tuple[str, str, list[float]]]
    ) -> None:
        if not items:
            return
        try:
            await self._entities.bulk_write(
                [
                    UpdateOne(
                        {"collection": collection, "key": key},
                        {"$set": {"embedding_text": text,
                                  "embedding": [float(x) for x in embedding]}},
                    )
                    for key, text, embedding in items
                ],
                ordered=False,
            )  # fmt: skip
        except PyMongoError as exc:
            raise GraphStoreError(f"Entity embedding write failed: {exc}") from exc

    async def set_entity_properties(self, collection: str, properties: dict[str, dict]) -> None:
        operations = []
        for key, values in properties.items():
            safe = {k: v for k, v in values.items() if _PROPERTY.match(k)}
            if safe:
                operations.append(UpdateOne({"collection": collection, "key": key}, {"$set": safe}))
        if operations:
            await self._entities.bulk_write(operations, ordered=False)

    async def merge_entities(self, collection: str, keep: str, merge: list[str]) -> int:
        """G-17/G-33: re-point a duplicate's edges to `keep`, then delete the duplicate."""
        merged = 0
        for duplicate in merge:
            if duplicate == keep:
                continue
            edges = self._relations.find(
                {"collection": collection, "$or": [{"src": duplicate}, {"dst": duplicate}]}
            )
            operations = []
            async for edge in edges:
                src = keep if edge["src"] == duplicate else edge["src"]
                dst = keep if edge["dst"] == duplicate else edge["dst"]
                if src == dst:
                    continue
                operations.append(
                    UpdateOne(
                        {"collection": collection, "src": src, "dst": dst, "type": edge["type"],
                         "chunk_id": edge.get("chunk_id", "")},
                        {"$set": {k: edge.get(k) for k in
                                  ("description", "source", "acl", "weight")}},
                        upsert=True,
                    )
                )  # fmt: skip
            if operations:
                await self._relations.bulk_write(operations, ordered=False)
            await self._relations.delete_many(
                {"collection": collection, "$or": [{"src": duplicate}, {"dst": duplicate}]}
            )
            mentions = {"collection": collection, "entity_keys": duplicate}
            await self._chunks.update_many(mentions, {"$addToSet": {"entity_keys": keep}})
            await self._chunks.update_many(mentions, {"$pull": {"entity_keys": duplicate}})
            await self._claims.update_many(
                {"collection": collection, "entity_key": duplicate}, {"$set": {"entity_key": keep}}
            )
            await self._entities.delete_one({"collection": collection, "key": duplicate})
            merged += 1
        return merged

    # -- reads ----------------------------------------------------------------------------
    async def find_entities(
        self, terms: list[str], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        found: dict[str, Entity] = {}
        for term in terms:
            if not term:
                continue
            # Over-fetch: visibility is checked next, on the chunks mentioning the entity.
            cursor = self._entities.find(
                {
                    "collection": filters.collection,
                    "$or": [
                        {"key": term},
                        {"key": {"$regex": re.escape(term)}},
                        {"alias_keys": term},
                    ],
                },
                _ENTITY_FIELDS,
                limit=int(limit) * 3,
            )
            candidates = [doc async for doc in cursor]
            visible = await self._visible_keys(
                filters.collection, [d["key"] for d in candidates], filters
            )
            for doc in candidates:
                if doc["key"] not in visible:
                    continue
                entity = _entity(doc)
                entity.score = (
                    1.0 if entity.key == term else SequenceMatcher(None, entity.key, term).ratio()
                )
                current = found.get(entity.key)
                if current is None or entity.score > current.score:
                    found[entity.key] = entity
        ordered = sorted(found.values(), key=lambda e: (-e.score, -e.rank, len(e.key)))
        return ordered[:limit]

    async def similar_entities(
        self, embedding: list[float], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        fetch = int(limit) * 3
        try:
            cursor = await self._entities.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": self._entity_vector_index,
                            "path": "embedding",
                            "queryVector": [float(x) for x in embedding],
                            "numCandidates": min(max(fetch * 10, 100), 10_000),
                            "limit": fetch,
                            "filter": {"collection": filters.collection},
                        }
                    },
                    {"$project": {**_ENTITY_FIELDS, "score": {"$meta": "vectorSearchScore"}}},
                ]
            )
            docs = await cursor.to_list()
        except OperationFailure as exc:
            logger.warning("entity_similarity_unavailable", error=str(exc))
            return []
        visible = await self._visible_keys(filters.collection, [d["key"] for d in docs], filters)
        entities = []
        for doc in docs:
            if doc["key"] in visible:
                entity = _entity(doc)
                entity.score = 2.0 * float(doc.get("score", 0.5)) - 1.0  # Atlas -> cosine
                entities.append(entity)
        return sorted(entities, key=lambda e: -e.score)[:limit]

    def _facts(self, edges: list[dict], names: dict[str, str]) -> list[GraphFact]:
        return [
            GraphFact(
                names.get(edge["src"], edge["src"]),
                edge.get("type") or "RELATED_TO",
                names.get(edge["dst"], edge["dst"]),
                edge.get("description") or "",
                edge["src"],
                edge["dst"],
                edge.get("chunk_id") or "",
                float(edge.get("weight") or 1.0),
                list(edge.get("acl") or []),
            )
            for edge in edges
        ]

    async def neighborhood(
        self,
        entity_keys: list[str],
        filters: RetrievalFilters,
        hops: int,
        limit: int,
        relation_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[GraphFact]:
        edges: dict[tuple[str, str, str], dict] = {}
        frontier, seen = set(entity_keys), set(entity_keys)
        for _ in range(hops):
            if not frontier or len(edges) >= limit:
                break
            keys = sorted(frontier)
            query: dict[str, Any] = {
                "collection": filters.collection,
                "$or": [{"src": {"$in": keys}}, {"dst": {"$in": keys}}],
                "weight": {"$gte": float(min_weight)},
                **_acl_clause(filters),
            }
            if relation_types:
                query["type"] = {"$in": list(relation_types)}
            next_frontier: set[str] = set()
            async for edge in self._relations.find(query, limit=int(limit) * 3):
                edges.setdefault((edge["src"], edge["type"], edge["dst"]), edge)
                next_frontier.update({edge["src"], edge["dst"]} - seen)
            seen |= next_frontier
            frontier = next_frontier
        kept = list(edges.values())[:limit]
        keys = {k for edge in kept for k in (edge["src"], edge["dst"])}
        names = await self._names(filters.collection, keys)
        return self._facts(kept, names)

    async def chunks_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[VectorSearchResult]:
        if not entity_keys:
            return []
        cursor = await self._chunks.aggregate(
            [
                {"$match": {"collection": filters.collection,
                            "entity_keys": {"$in": entity_keys},
                            **_acl_clause(filters)}},
                {"$addFields": {"hits": {"$size": {"$setIntersection": [
                    "$entity_keys", entity_keys]}}}},
                {"$sort": {"hits": DESCENDING, "_id": ASCENDING}},
                {"$limit": int(limit)},
                {"$project": {"source": 1, "content": 1, "acl": 1, "doc_id": 1, "title": 1,
                              "hits": 1}},
            ]
        )  # fmt: skip
        return [
            VectorSearchResult(
                chunk_id=str(doc["_id"]),
                content=doc.get("content") or "",
                score=float(doc.get("hits", 0)) / len(entity_keys),
                source=doc.get("source") or "",
                metadata={"retriever": "graph"},
                acl=list(doc.get("acl") or []),
                doc_id=doc.get("doc_id") or "",
                title=doc.get("title") or "",
            )
            async for doc in cursor
        ]

    async def entities_for_chunks(
        self, chunk_ids: list[str], filters: RetrievalFilters
    ) -> list[Entity]:
        if not chunk_ids:
            return []
        keys = await self._chunks.distinct(
            "entity_keys",
            {"_id": {"$in": chunk_ids}, "collection": filters.collection,
             **_acl_clause(filters)},
        )  # fmt: skip
        docs = await self._entities_by_key(filters.collection, keys)
        return [_entity(doc) for doc in docs.values()]

    async def claims_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[tuple[Claim, str]]:
        if not entity_keys:
            return []
        cursor = self._claims.find(
            {"collection": filters.collection, "entity_key": {"$in": entity_keys},
             **_acl_clause(filters)},
            limit=int(limit),
        )  # fmt: skip
        rows = [doc async for doc in cursor]
        names = await self._names(filters.collection, {r["entity_key"] for r in rows})
        return [
            (
                Claim(names.get(r["entity_key"], r["entity_key"]), r.get("text") or "",
                      r.get("date") or None),
                r.get("chunk_id") or "",
            )
            for r in rows
        ]  # fmt: skip

    async def export_graph(self, collection: str, acl_scope: frozenset[str] | None) -> GraphExport:
        export = GraphExport()
        visible = set(
            await self._chunks.distinct(
                "entity_keys", {"collection": collection, **_acl_clause(acl_scope)}
            )
        )
        async for doc in self._entities.find({"collection": collection}, _ENTITY_FIELDS):
            if doc["key"] in visible:
                export.nodes[doc["key"]] = _entity(doc)
        names = {key: entity.name for key, entity in export.nodes.items()}
        edges = [
            edge
            async for edge in self._relations.find({"collection": collection})
            if is_allowed(edge.get("acl") or [], acl_scope)
            and edge["src"] in export.nodes
            and edge["dst"] in export.nodes
        ]
        export.edges = self._facts(edges, names)
        return export

    async def quality_report(self, collection: str) -> dict:
        """G-24: counts, orphans, likely duplicates and a random triple sample for review."""
        scope = {"collection": collection}
        names = {"Entity": "entities", "Chunk": "chunks", "Document": "documents",
                 "Section": "sections", "Claim": "claims"}  # fmt: skip
        counts = {
            label: await self._col(names[label]).count_documents(scope) for label in _COUNTED_LABELS
        }
        connected = set(await self._relations.distinct("src", scope)) | set(
            await self._relations.distinct("dst", scope)
        )
        by_type: dict[str, int] = {}
        keys: list[str] = []
        orphans = 0
        async for doc in self._entities.find(scope, {"key": 1, "type": 1}):
            keys.append(doc["key"])
            by_type[doc.get("type") or "?"] = by_type.get(doc.get("type") or "?", 0) + 1
            orphans += doc["key"] not in connected
        keys.sort()
        duplicates = [
            (a, b)
            for a, b in zip(keys, keys[1:], strict=False)
            if SequenceMatcher(None, a, b).ratio() >= 0.9
        ][:50]
        sample = [edge async for edge in self._relations.find(scope, limit=500)]
        keys = {k for edge in sample for k in (edge["src"], edge["dst"])}
        entity_names = await self._names(collection, keys)
        triples = [
            (entity_names.get(e["src"], e["src"]), e.get("type"),
             entity_names.get(e["dst"], e["dst"]), e.get("description") or "")
            for e in sample
        ]  # fmt: skip
        return {
            "counts": counts,
            "entities_by_type": by_type,
            "orphan_entities": orphans,
            "duplicate_candidates": duplicates,
            "triple_sample": random.sample(triples, min(20, len(triples))),  # noqa: S311
        }

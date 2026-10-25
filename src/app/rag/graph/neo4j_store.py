"""Graph store on Neo4j (Aura): the graph model of base.py as native nodes and relationships.

  (:Document {uid, collection, doc_id, title, source})
  (:Section  {uid, collection, key, name, doc_id, source})-[:PART_OF]->(:Document)
  (:Chunk    {chunk_id, collection, source, content, acl, account_id, ...})-[:PART_OF]->(:Section)
  (:Chunk)-[:NEXT]->(:Chunk)                                    reading order (G-20)
  (:Chunk)-[:MENTIONS]->(:Entity {uid, collection, key, name, type, rank, community_<n>})
  (:Entity)-[:RELATED {type, chunk_id, weight, acl, account_id, ...}]->(:Entity)   one per
                                                                (type, chunk) (G-11)
  (:Claim {uid, text, date, chunk_id, acl, account_id})-[:ABOUT]->(:Entity)        (G-12)

`uid` is "<collection>|<key>": a single-property uniqueness constraint per label, so the
schema works on every Aura tier. Entity description embeddings sit on the Entity node with
a Neo4j vector index for entity linking (G-25, G-37).

ACL (G-51) is enforced in the queries, as in the Mongo store: an entity is visible when a
chunk the caller may read mentions it; relations, claims and chunks carry their chunk's
ACL. Cypher (G-43..G-45) runs natively: templates and validated text-to-Cypher.
"""

import hashlib
import random
import re
from difflib import SequenceMatcher
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, Query, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError

from app.core.exceptions import GraphStoreError
from app.core.logging import get_logger
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

_INDEX_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,60}$")
_PROPERTY = re.compile(r"^(rank|community_\d{1,2})$")
_COUNTED_LABELS = ("Entity", "Chunk", "Document", "Section", "Claim")
_ENTITY_RETURN = (
    "e.key AS key, e.name AS name, e.type AS type, e.description AS description, "
    "e.rank AS rank, e.community_0 AS community_0, e.external_id AS external_id"
)
_FACT_RETURN = (
    "startNode(r).key AS src, endNode(r).key AS dst, startNode(r).name AS src_name, "
    "endNode(r).name AS dst_name, r.type AS type, r.description AS description, "
    "r.chunk_id AS chunk_id, r.weight AS weight, r.acl AS acl"
)
_SCHEMA = (
    "CREATE CONSTRAINT rag_document_uid IF NOT EXISTS FOR (n:Document) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT rag_section_uid IF NOT EXISTS FOR (n:Section) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT rag_chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT rag_entity_uid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT rag_claim_uid IF NOT EXISTS FOR (n:Claim) REQUIRE n.uid IS UNIQUE",
    "CREATE INDEX rag_chunk_source IF NOT EXISTS FOR (n:Chunk) ON (n.collection, n.source)",
    "CREATE INDEX rag_chunk_prev IF NOT EXISTS FOR (n:Chunk) ON (n.prev_id)",
    "CREATE INDEX rag_entity_collection IF NOT EXISTS FOR (n:Entity) ON (n.collection)",
    "CREATE TEXT INDEX rag_entity_key_text IF NOT EXISTS FOR (n:Entity) ON (n.key)",
    "CREATE INDEX rag_section_source IF NOT EXISTS FOR (n:Section) ON (n.collection, n.source)",
    "CREATE INDEX rag_document_source IF NOT EXISTS FOR (n:Document) ON (n.collection, n.source)",
    "CREATE INDEX rag_claim_source IF NOT EXISTS FOR (n:Claim) ON (n.collection, n.source)",
    "CREATE INDEX rag_related_source IF NOT EXISTS "
    "FOR ()-[r:RELATED]-() ON (r.collection, r.source)",
)

# Writes for one chunk, run in a single transaction.
_WRITE_CHUNK = """
MERGE (d:Document {uid: $doc_uid})
SET d.collection = $collection, d.doc_id = $doc_id, d.title = $title, d.source = $source
MERGE (s:Section {uid: $section_uid})
SET s.collection = $collection, s.key = $section_key, s.name = $section,
    s.doc_id = $doc_id, s.source = $source
MERGE (s)-[:PART_OF]->(d)
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.collection = $collection, c.source = $source, c.content = $content,
    c.chunk_index = $chunk_index, c.acl = $acl, c.account_id = $account_id,
    c.doc_id = $doc_id, c.title = $title, c.section_key = $section_key, c.prev_id = $prev_id
MERGE (c)-[:PART_OF]->(s)
WITH c
OPTIONAL MATCH (p:Chunk {chunk_id: $prev_id})
FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (p)-[:NEXT]->(c))
WITH c
OPTIONAL MATCH (n:Chunk {prev_id: $chunk_id})
FOREACH (_ IN CASE WHEN n IS NULL THEN [] ELSE [1] END | MERGE (c)-[:NEXT]->(n))
"""
_WRITE_ENTITIES = """
MATCH (c:Chunk {chunk_id: $chunk_id})
UNWIND $entities AS ent
MERGE (e:Entity {uid: ent.uid})
ON CREATE SET e.description = ent.description, e.alias_keys = ent.alias_keys,
              e.external_id = ent.external_id
SET e.collection = $collection, e.key = ent.key, e.name = ent.name, e.type = ent.type
MERGE (c)-[:MENTIONS]->(e)
"""
# Like the Mongo store: relations / claims only between entities that exist.
_WRITE_RELATIONS = """
UNWIND $relations AS rel
MATCH (a:Entity {uid: rel.src_uid}), (b:Entity {uid: rel.dst_uid})
MERGE (a)-[r:RELATED {type: rel.type, chunk_id: $chunk_id}]->(b)
SET r.collection = $collection, r.description = rel.description, r.source = $source,
    r.acl = $acl, r.account_id = $account_id, r.weight = rel.weight
"""
_WRITE_CLAIMS = """
UNWIND $claims AS claim
MATCH (e:Entity {uid: claim.entity_uid})
MERGE (cl:Claim {uid: claim.uid})
SET cl.collection = $collection, cl.entity_key = claim.entity_key, cl.text = claim.text,
    cl.date = claim.date, cl.chunk_id = $chunk_id, cl.source = $source, cl.acl = $acl,
    cl.account_id = $account_id
MERGE (cl)-[:ABOUT]->(e)
"""


def _uid(collection: str, key: str) -> str:
    return f"{collection}|{key}"


def _entity(row: dict[str, Any]) -> Entity:
    community = row.get("community_0")
    return Entity(
        name=row.get("name") or row.get("key") or "",
        type=row.get("type") or "Concept",
        description=row.get("description") or "",
        rank=float(row.get("rank") or 0.0),
        community=community if isinstance(community, int) else None,
        external_id=row.get("external_id") or "",
    )


def _access(alias: str, principals: frozenset[str] | None, account_id: str | None) -> str:
    """WHERE fragment restricting `alias` (a chunk, relation or claim) to the caller's ACL and
    account; binds $principals / $account_id. "true" when neither is enforced."""
    clauses = []
    if principals is not None:
        clauses.append(f"any(p IN coalesce({alias}.acl, []) WHERE p IN $principals)")
    if account_id is not None:
        clauses.append(f"coalesce({alias}.account_id, '') = $account_id")
    return " AND ".join(clauses) or "true"


def _access_params(principals: frozenset[str] | None, account_id: str | None) -> dict[str, Any]:
    return {
        "principals": sorted(principals) if principals is not None else [],
        "account_id": account_id or "",
    }


class Neo4jGraphStore(GraphStore):
    supports_cypher = True

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        entity_vector_index: str = "rag_entity_embedding",
        embedding_dim: int = 1536,
        query_timeout_seconds: float = 30.0,
    ) -> None:
        if not uri or not password:
            raise GraphStoreError("NEO4J_URI and NEO4J_PASSWORD are not configured")
        if not _INDEX_NAME.match(entity_vector_index):
            raise GraphStoreError(f"Invalid entity vector index name: {entity_vector_index!r}")
        self._uri = uri
        self._auth = (username, password)
        self._database = database
        self._entity_vector_index = entity_vector_index
        self._embedding_dim = int(embedding_dim)
        self._timeout = float(query_timeout_seconds)
        self._driver: AsyncDriver | None = None

    # -- plumbing -------------------------------------------------------------------------
    @property
    def driver(self) -> AsyncDriver:
        # Created lazily, inside the running event loop.
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=self._auth, user_agent="knowledge-service"
            )
        return self._driver

    async def _run(
        self, cypher: str, params: dict[str, Any] | None = None, *, read: bool = False
    ) -> list[dict[str, Any]]:
        try:
            result = await self.driver.execute_query(
                Query(cypher, timeout=self._timeout),
                params or {},
                database_=self._database,
                routing_=RoutingControl.READ if read else RoutingControl.WRITE,
            )
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Graph query failed: {exc}") from exc
        return [record.data() for record in result.records]

    async def _read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        return await self._run(cypher, params, read=True)

    async def _visible_keys(
        self, collection: str, keys: list[str], filters: RetrievalFilters
    ) -> set[str]:
        """Entity keys mentioned by at least one chunk the caller may read."""
        if not keys:
            return set()
        rows = await self._read(
            f"""
            MATCH (c:Chunk {{collection: $collection}})-[:MENTIONS]->(e:Entity)
            WHERE e.uid IN $uids AND {_access("c", filters.principals, filters.account_id)}
            RETURN DISTINCT e.key AS key
            """,
            {
                "collection": collection,
                "uids": [_uid(collection, k) for k in keys],
                **_access_params(filters.principals, filters.account_id),
            },
        )
        return {row["key"] for row in rows}

    async def _names(self, collection: str, keys: set[str]) -> dict[str, str]:
        if not keys:
            return {}
        rows = await self._read(
            "MATCH (e:Entity) WHERE e.uid IN $uids RETURN e.key AS key, e.name AS name",
            {"uids": [_uid(collection, k) for k in keys]},
        )
        return {row["key"]: row.get("name") or row["key"] for row in rows}

    # -- lifecycle ------------------------------------------------------------------------
    async def ensure_schema(self) -> None:
        for statement in _SCHEMA:
            await self._run(statement)
        await self._ensure_entity_vector_index()
        logger.info("graph_schema_ready", provider="neo4j", database=self._database)

    async def _ensure_entity_vector_index(self) -> None:
        """Vector index for entity linking (G-25). Skipped when the server lacks one."""
        try:
            await self._run(
                f"CREATE VECTOR INDEX {self._entity_vector_index} IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {self._embedding_dim}, "
                "`vector.similarity_function`: 'cosine'}}"
            )
        except GraphStoreError as exc:
            logger.warning("entity_embeddings_unavailable", error=str(exc))

    async def ping(self) -> None:
        try:
            await self.driver.verify_connectivity()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Graph DB is unreachable: {exc}") from exc

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    # -- writes ---------------------------------------------------------------------------
    async def upsert_chunk_graph(self, chunk: ChunkGraph) -> None:
        collection = chunk.collection
        section = " > ".join(chunk.section_path) or chunk.title
        section_key = f"{chunk.doc_id}|{section}"
        known = {e.key for e in chunk.entities}
        base = {
            "collection": collection,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "acl": list(chunk.acl),
            "account_id": chunk.account_id,
        }
        entities = [
            {
                "uid": _uid(collection, e.key),
                "key": e.key,
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "alias_keys": sorted({entity_key(a) for a in e.aliases}),
                "external_id": e.external_id,
            }
            for e in chunk.entities
        ]
        relations = [
            {
                "src_uid": _uid(collection, entity_key(r.source)),
                "dst_uid": _uid(collection, entity_key(r.target)),
                "type": r.type,
                "description": r.description,
                "weight": float(r.weight),
            }
            for r in chunk.relations
        ]
        claims = []
        for claim in chunk.claims:
            subject = entity_key(claim.subject)
            digest = hashlib.sha1(  # noqa: S324 - identity, not security
                f"{chunk.chunk_id}|{subject}|{claim.text}".encode()
            ).hexdigest()
            claims.append(
                {
                    "uid": _uid(collection, digest),
                    "entity_uid": _uid(collection, subject),
                    "entity_key": subject,
                    "text": claim.text,
                    "date": claim.date or "",
                }
            )

        async def write(tx) -> None:
            await tx.run(
                _WRITE_CHUNK,
                {
                    **base,
                    "doc_uid": _uid(collection, chunk.doc_id),
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section_uid": _uid(collection, section_key),
                    "section_key": section_key,
                    "section": section,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "prev_id": chunk.prev_id,
                },
            )
            if entities:
                await tx.run(_WRITE_ENTITIES, {**base, "entities": entities})
            if relations:
                await tx.run(_WRITE_RELATIONS, {**base, "relations": relations})
            if claims:
                await tx.run(_WRITE_CLAIMS, {**base, "claims": claims})

        try:
            async with self.driver.session(database=self._database) as session:
                await session.execute_write(write)
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Graph write failed: {exc}") from exc
        logger.debug("graph_chunk_written", chunk_id=chunk.chunk_id, entities=len(known))

    async def _delete_in_batches(self, match: str, params: dict[str, Any]) -> None:
        """DETACH DELETE the nodes `match` binds as n, 1000 per transaction (auto-commit)."""
        try:
            async with self.driver.session(database=self._database) as session:
                result = await session.run(
                    f"{match} CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 1000 ROWS",
                    params,
                )
                await result.consume()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Graph delete failed: {exc}") from exc

    async def _count(self, match: str, params: dict[str, Any]) -> int:
        rows = await self._read(f"{match} RETURN count(n) AS n", params)
        return int(rows[0]["n"]) if rows else 0

    async def delete_by_source(self, source: str, collection: str) -> int:
        scope = {"collection": collection, "source": source}
        await self._run(
            "MATCH ()-[r:RELATED {collection: $collection, source: $source}]->() DELETE r", scope
        )
        chunks = "MATCH (n:Chunk {collection: $collection, source: $source})"
        deleted = await self._count(chunks, scope)
        for match in (
            "MATCH (n:Claim {collection: $collection, source: $source})",
            chunks,
            "MATCH (n:Section {collection: $collection, source: $source})",
            "MATCH (n:Document {collection: $collection, source: $source})",
        ):
            await self._delete_in_batches(match, scope)
        await self._delete_orphan_entities(collection)
        return deleted

    async def delete_collection(self, collection: str) -> int:
        deleted = 0
        for label in ("Claim", "Chunk", "Section", "Document", "Entity"):
            match = f"MATCH (n:{label} {{collection: $collection}})"
            deleted += await self._count(match, {"collection": collection})
            await self._delete_in_batches(match, {"collection": collection})
        return deleted

    async def _delete_orphan_entities(self, collection: str) -> None:
        """G-32: entities no chunk mentions any more are removed with their edges and claims."""
        await self._delete_in_batches(
            "MATCH (e:Entity {collection: $collection}) "
            "WHERE NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(e) } "
            "OPTIONAL MATCH (cl:Claim)-[:ABOUT]->(e) "
            "UNWIND [x IN [e, cl] WHERE x IS NOT NULL] AS n WITH DISTINCT n",
            {"collection": collection},
        )

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        params = {"collection": collection, "source": source, "acl": list(acl)}
        rows = await self._run(
            "MATCH (c:Chunk {collection: $collection, source: $source}) SET c.acl = $acl "
            "RETURN count(c) AS n",
            params,
        )
        await self._run(
            "MATCH ()-[r:RELATED {collection: $collection, source: $source}]->() SET r.acl = $acl",
            params,
        )
        await self._run(
            "MATCH (cl:Claim {collection: $collection, source: $source}) SET cl.acl = $acl", params
        )
        return int(rows[0]["n"]) if rows else 0

    async def upsert_entity_embeddings(
        self, collection: str, items: list[tuple[str, str, list[float]]]
    ) -> None:
        if not items:
            return
        await self._run(
            "UNWIND $items AS item MATCH (e:Entity {uid: item.uid}) "
            "SET e.embedding_text = item.text, e.embedding = item.embedding",
            {
                "items": [
                    {"uid": _uid(collection, key), "text": text,
                     "embedding": [float(x) for x in embedding]}
                    for key, text, embedding in items
                ]
            },
        )  # fmt: skip

    async def set_entity_properties(self, collection: str, properties: dict[str, dict]) -> None:
        items = []
        for key, values in properties.items():
            safe = {k: v for k, v in values.items() if _PROPERTY.match(k)}
            if safe:
                items.append({"uid": _uid(collection, key), "props": safe})
        if items:
            await self._run(
                "UNWIND $items AS item MATCH (e:Entity {uid: item.uid}) SET e += item.props",
                {"items": items},
            )

    async def merge_entities(self, collection: str, keep: str, merge: list[str]) -> int:
        """G-17/G-33: re-point a duplicate's edges, mentions and claims to `keep`, then delete
        the duplicate."""
        merged = 0
        for duplicate in merge:
            if duplicate == keep:
                continue
            params = {
                "keep": _uid(collection, keep),
                "dup": _uid(collection, duplicate),
                "keep_key": keep,
            }
            rows = await self._run(
                """
                MATCH (k:Entity {uid: $keep}), (d:Entity {uid: $dup})
                CALL (k, d) {
                  MATCH (d)-[r:RELATED]->(x) WHERE x <> k
                  MERGE (k)-[n:RELATED {type: r.type, chunk_id: r.chunk_id}]->(x)
                  SET n += properties(r)
                }
                CALL (k, d) {
                  MATCH (x)-[r:RELATED]->(d) WHERE x <> k
                  MERGE (x)-[n:RELATED {type: r.type, chunk_id: r.chunk_id}]->(k)
                  SET n += properties(r)
                }
                CALL (k, d) {
                  MATCH (c:Chunk)-[:MENTIONS]->(d) MERGE (c)-[:MENTIONS]->(k)
                }
                CALL (k, d) {
                  MATCH (cl:Claim)-[:ABOUT]->(d) MERGE (cl)-[:ABOUT]->(k)
                  SET cl.entity_key = $keep_key
                }
                DETACH DELETE d
                RETURN 1 AS merged
                """,
                params,
            )
            merged += len(rows)
        return merged

    # -- reads ----------------------------------------------------------------------------
    async def find_entities(
        self, terms: list[str], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        found: dict[str, Entity] = {}
        for term in terms:
            if not term:
                continue
            rows = await self._read(
                f"""
                MATCH (e:Entity {{collection: $collection}})
                WHERE e.key = $term OR e.key CONTAINS $term OR $term IN coalesce(e.alias_keys, [])
                WITH e LIMIT $fetch
                WHERE EXISTS {{
                  MATCH (c:Chunk)-[:MENTIONS]->(e)
                  WHERE {_access("c", filters.principals, filters.account_id)}
                }}
                RETURN {_ENTITY_RETURN}
                """,
                {
                    "collection": filters.collection,
                    "term": term,
                    "fetch": int(limit) * 3,
                    **_access_params(filters.principals, filters.account_id),
                },
            )
            for row in rows:
                entity = _entity(row)
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
            rows = await self._read(
                f"""
                CALL db.index.vector.queryNodes($index, $candidates, $embedding)
                YIELD node AS e, score
                WHERE e.collection = $collection
                WITH e, score LIMIT $fetch
                RETURN {_ENTITY_RETURN}, score
                """,
                {
                    "index": self._entity_vector_index,
                    "candidates": min(max(fetch * 10, 100), 10_000),
                    "embedding": [float(x) for x in embedding],
                    "collection": filters.collection,
                    "fetch": fetch,
                },
            )
        except GraphStoreError as exc:
            logger.warning("entity_similarity_unavailable", error=str(exc))
            return []
        visible = await self._visible_keys(filters.collection, [r["key"] for r in rows], filters)
        entities = []
        for row in rows:
            if row["key"] in visible:
                entity = _entity(row)
                entity.score = 2.0 * float(row.get("score", 0.5)) - 1.0  # (1 + cos) / 2 -> cos
                entities.append(entity)
        return sorted(entities, key=lambda e: -e.score)[:limit]

    @staticmethod
    def _facts(rows: list[dict[str, Any]]) -> list[GraphFact]:
        return [
            GraphFact(
                row.get("src_name") or row["src"],
                row.get("type") or "RELATED_TO",
                row.get("dst_name") or row["dst"],
                row.get("description") or "",
                row["src"],
                row["dst"],
                row.get("chunk_id") or "",
                float(row.get("weight") or 1.0),
                list(row.get("acl") or []),
            )
            for row in rows
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
        # Hop by hop, so ACL and weight pruning apply to every edge on the way (G-38..G-40).
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        frontier, seen = set(entity_keys), set(entity_keys)
        for _ in range(hops):
            if not frontier or len(edges) >= limit:
                break
            rows = await self._read(
                f"""
                MATCH (a:Entity)-[r:RELATED]-(:Entity)
                WHERE a.uid IN $uids AND coalesce(r.weight, 1.0) >= $min_weight
                  AND ($types IS NULL OR r.type IN $types)
                  AND {_access("r", filters.principals, filters.account_id)}
                RETURN DISTINCT {_FACT_RETURN}
                LIMIT $fetch
                """,
                {
                    "uids": [_uid(filters.collection, k) for k in sorted(frontier)],
                    "min_weight": float(min_weight),
                    "types": list(relation_types) if relation_types else None,
                    "fetch": int(limit) * 3,
                    **_access_params(filters.principals, filters.account_id),
                },
            )
            next_frontier: set[str] = set()
            for row in rows:
                edges.setdefault((row["src"], row["type"], row["dst"]), row)
                next_frontier.update({row["src"], row["dst"]} - seen)
            seen |= next_frontier
            frontier = next_frontier
        return self._facts(list(edges.values())[:limit])

    async def chunks_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[VectorSearchResult]:
        if not entity_keys:
            return []
        rows = await self._read(
            f"""
            MATCH (c:Chunk {{collection: $collection}})-[:MENTIONS]->(e:Entity)
            WHERE e.uid IN $uids AND {_access("c", filters.principals, filters.account_id)}
            WITH c, count(DISTINCT e) AS hits
            ORDER BY hits DESC, c.chunk_id ASC
            LIMIT $limit
            RETURN c.chunk_id AS chunk_id, c.source AS source, c.content AS content,
                   c.acl AS acl, c.doc_id AS doc_id, c.title AS title, hits
            """,
            {
                "collection": filters.collection,
                "uids": [_uid(filters.collection, k) for k in entity_keys],
                "limit": int(limit),
                **_access_params(filters.principals, filters.account_id),
            },
        )
        return [
            VectorSearchResult(
                chunk_id=str(row["chunk_id"]),
                content=row.get("content") or "",
                score=float(row.get("hits", 0)) / len(entity_keys),
                source=row.get("source") or "",
                metadata={"retriever": "graph"},
                acl=list(row.get("acl") or []),
                doc_id=row.get("doc_id") or "",
                title=row.get("title") or "",
            )
            for row in rows
        ]

    async def entities_for_chunks(
        self, chunk_ids: list[str], filters: RetrievalFilters
    ) -> list[Entity]:
        if not chunk_ids:
            return []
        rows = await self._read(
            f"""
            MATCH (c:Chunk {{collection: $collection}})-[:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IN $chunk_ids
              AND {_access("c", filters.principals, filters.account_id)}
            WITH DISTINCT e
            RETURN {_ENTITY_RETURN}
            """,
            {
                "collection": filters.collection,
                "chunk_ids": list(chunk_ids),
                **_access_params(filters.principals, filters.account_id),
            },
        )
        return [_entity(row) for row in rows]

    async def claims_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[tuple[Claim, str]]:
        if not entity_keys:
            return []
        rows = await self._read(
            f"""
            MATCH (cl:Claim)-[:ABOUT]->(e:Entity)
            WHERE e.uid IN $uids AND {_access("cl", filters.principals, filters.account_id)}
            RETURN e.name AS name, e.key AS key, cl.text AS text, cl.date AS date,
                   cl.chunk_id AS chunk_id
            LIMIT $limit
            """,
            {
                "uids": [_uid(filters.collection, k) for k in entity_keys],
                "limit": int(limit),
                **_access_params(filters.principals, filters.account_id),
            },
        )
        return [
            (
                Claim(row.get("name") or row["key"], row.get("text") or "",
                      row.get("date") or None),
                row.get("chunk_id") or "",
            )
            for row in rows
        ]  # fmt: skip

    async def export_graph(self, collection: str, acl_scope: frozenset[str] | None) -> GraphExport:
        export = GraphExport()
        rows = await self._read(
            f"""
            MATCH (e:Entity {{collection: $collection}})
            WHERE EXISTS {{
              MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE {_access("c", acl_scope, None)}
            }}
            RETURN {_ENTITY_RETURN}
            """,
            {"collection": collection, **_access_params(acl_scope, None)},
        )
        for row in rows:
            export.nodes[row["key"]] = _entity(row)
        edges = await self._read(
            f"""
            MATCH (:Entity {{collection: $collection}})-[r:RELATED]->(:Entity)
            RETURN {_FACT_RETURN}
            """,
            {"collection": collection},
        )
        export.edges = self._facts(
            [
                row
                for row in edges
                if is_allowed(row.get("acl") or [], acl_scope)
                and row["src"] in export.nodes
                and row["dst"] in export.nodes
            ]
        )
        return export

    async def run_cypher(self, query: str, params: dict, columns: tuple[str, ...]) -> list[tuple]:
        """G-43: a query already validated by cypher.validate_cypher, on a read route."""
        try:
            result = await self.driver.execute_query(
                Query(query, timeout=self._timeout),
                params,
                database_=self._database,
                routing_=RoutingControl.READ,
            )
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreError(f"Cypher query failed: {exc}") from exc
        return [tuple(record.values())[: len(columns) or None] for record in result.records]

    async def quality_report(self, collection: str) -> dict:
        """G-24: counts, orphans, likely duplicates and a random triple sample for review."""
        scope = {"collection": collection}
        counts = {}
        for label in _COUNTED_LABELS:
            rows = await self._read(
                f"MATCH (n:{label} {{collection: $collection}}) RETURN count(n) AS n", scope
            )
            counts[label] = int(rows[0]["n"]) if rows else 0
        entities = await self._read(
            "MATCH (e:Entity {collection: $collection}) "
            "RETURN e.key AS key, e.type AS type, EXISTS { (e)-[:RELATED]-() } AS connected",
            scope,
        )
        by_type: dict[str, int] = {}
        for row in entities:
            by_type[row.get("type") or "?"] = by_type.get(row.get("type") or "?", 0) + 1
        orphans = sum(1 for row in entities if not row["connected"])
        keys = sorted(row["key"] for row in entities)
        duplicates = [
            (a, b)
            for a, b in zip(keys, keys[1:], strict=False)
            if SequenceMatcher(None, a, b).ratio() >= 0.9
        ][:50]
        sample = await self._read(
            "MATCH (:Entity {collection: $collection})-[r:RELATED]->(:Entity) "
            f"RETURN {_FACT_RETURN} LIMIT 500",
            scope,
        )
        triples = [
            (row.get("src_name") or row["src"], row.get("type"),
             row.get("dst_name") or row["dst"], row.get("description") or "")
            for row in sample
        ]  # fmt: skip
        return {
            "counts": counts,
            "entities_by_type": by_type,
            "orphan_entities": orphans,
            "duplicate_candidates": duplicates,
            "triple_sample": random.sample(triples, min(20, len(triples))),  # noqa: S311
        }

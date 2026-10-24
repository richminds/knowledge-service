"""Graph model and store contract (Part C).

Graph model (all nodes carry `collection`; chunk-derived items carry the chunk's ACL):
  lexical layer (G-20):  (:Document)<-[:PART_OF]-(:Section)<-[:PART_OF]-(:Chunk)-[:NEXT]->(:Chunk)
  entity layer:          (:Chunk)-[:MENTIONS]->(:Entity)-[:RELATED {type, weight}]->(:Entity)
  claims (G-12):         (:Claim {text, date})-[:ABOUT]->(:Entity)
  enrichment (G-27/30):  Entity.community_<level>, Entity.rank; community reports are
                         indexed in the vector store (kind=community, G-29)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.rag.types import PUBLIC, RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult


def entity_key(name: str) -> str:
    """Canonical identity for an entity: lower-case, single-spaced."""
    return re.sub(r"\s+", " ", name).strip().lower()


@dataclass
class Entity:
    name: str
    type: str = "Concept"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    external_id: str = ""  # master-data / knowledge-base id (G-18)
    rank: float = 0.0  # centrality (G-30)
    community: int | None = None  # finest community (G-27)
    score: float = 0.0  # match score when returned by entity linking (G-37)

    @property
    def key(self) -> str:
        return entity_key(self.name)


@dataclass
class Relation:
    source: str  # entity name
    target: str  # entity name
    type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0  # extraction confidence / strength (G-11)


@dataclass
class Claim:
    subject: str  # entity name
    text: str
    date: str | None = None  # ISO date when the claim is dated (G-12)


@dataclass
class ChunkGraph:
    """One chunk plus the entities/relations/claims extracted from it."""

    chunk_id: str
    source: str
    collection: str
    content: str
    chunk_index: int = 0
    acl: list[str] = field(default_factory=lambda: [PUBLIC])
    account_id: str = ""  # app account it was ingested from ("" = none)
    doc_id: str = ""
    title: str = ""
    section_path: list[str] = field(default_factory=list)  # lexical layer (G-20)
    prev_id: str = ""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)


@dataclass
class GraphFact:
    source_name: str
    relation: str
    target_name: str
    description: str = ""
    source_key: str = ""
    target_key: str = ""
    chunk_id: str = ""  # chunk the fact was extracted from (G-53: keeps facts citable)
    weight: float = 1.0
    acl: list[str] = field(default_factory=lambda: [PUBLIC])

    def render(self) -> str:
        detail = f": {self.description}" if self.description else ""
        return f"{self.source_name} -[{self.relation}]-> {self.target_name}{detail}"


@dataclass
class Community:
    community_id: str  # "<level>:<n>"
    level: int
    entity_keys: list[str]
    title: str = ""
    summary: str = ""
    rank: float = 0.0  # importance (size x centrality)


@dataclass
class GraphExport:
    """Entity graph for offline algorithms (communities, PageRank, PPR)."""

    nodes: dict[str, Entity] = field(default_factory=dict)  # key -> entity
    edges: list[GraphFact] = field(default_factory=list)


class GraphStore(ABC):
    """Abstraction over the graph database so providers are swappable."""

    # Whether run_cypher works; without it the cypher search mode falls back to local.
    supports_cypher: bool = False

    async def ensure_schema(self) -> None:  # noqa: B027 - optional hook
        """Create the graph, labels and indexes if missing."""

    async def ping(self) -> None:  # noqa: B027 - optional hook
        """Raise if the store is unreachable (readiness probe)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections on shutdown."""

    @abstractmethod
    async def upsert_chunk_graph(self, chunk: ChunkGraph) -> None:
        """Write the chunk (+ document/section/NEXT), its entities, relations and claims."""

    # Reads take RetrievalFilters and must enforce its ACL (G-51): an entity is visible only
    # when a chunk the caller may read mentions it; facts and chunks carry their chunk's ACL.
    @abstractmethod
    async def find_entities(
        self, terms: list[str], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        """Entities whose key or alias equals/contains any of `terms` (normalised)."""

    @abstractmethod
    async def neighborhood(
        self,
        entity_keys: list[str],
        filters: RetrievalFilters,
        hops: int,
        limit: int,
        relation_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[GraphFact]:
        """Relations reachable within `hops` (G-38, G-39), pruned by type/weight (G-40)."""

    @abstractmethod
    async def chunks_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[VectorSearchResult]:
        """Chunks that mention the entities; score = share of entities mentioned."""

    @abstractmethod
    async def delete_by_source(self, source: str, collection: str) -> int:
        """Delete a source's chunks (and the relations/claims extracted from them)."""

    @abstractmethod
    async def delete_collection(self, collection: str) -> int:
        """Delete every node in a collection."""

    # -- optional capabilities (stores may not support all of them) -------------------------
    async def entities_for_chunks(
        self, chunk_ids: list[str], filters: RetrievalFilters
    ) -> list[Entity]:
        """Entities mentioned by the chunks (vector -> graph hybrid, G-50)."""
        return []

    async def claims_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[tuple[Claim, str]]:
        """(claim, chunk_id) pairs about the entities (G-12)."""
        return []

    async def similar_entities(
        self, embedding: list[float], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        """Entity linking by description-embedding similarity (G-25, G-37)."""
        return []

    async def upsert_entity_embeddings(
        self, collection: str, items: list[tuple[str, str, list[float]]]
    ) -> None:
        """(entity key, text embedded, embedding) for entity linking (G-25)."""

    async def export_graph(self, collection: str, acl_scope: frozenset[str] | None) -> GraphExport:
        """Entity graph limited to facts visible to `acl_scope` (None = everything)."""
        return GraphExport()

    async def set_entity_properties(self, collection: str, properties: dict[str, dict]) -> None:
        """Write computed properties (rank, community_<level>) onto entities (G-27, G-30)."""

    async def merge_entities(self, collection: str, keep: str, merge: list[str]) -> int:
        """Merge duplicate entities into `keep`, re-pointing their edges (G-17, G-33)."""
        return 0

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        """Permissions of a source's chunks/relations changed (C-145)."""
        return 0

    async def run_cypher(self, query: str, params: dict, columns: tuple[str, ...]) -> list[tuple]:
        """Execute a validated read-only Cypher query (G-43)."""
        raise NotImplementedError("This graph store does not support Cypher")

    async def quality_report(self, collection: str) -> dict:
        """Counts, orphans, duplicate-name candidates, sample triples (G-24)."""
        return {}

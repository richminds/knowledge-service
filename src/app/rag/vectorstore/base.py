from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field

from app.rag.types import PUBLIC, RetrievalFilters

# Record kinds in the index:
#   chunk    - a retrievable passage (the default)
#   question - a generated question the chunk answers (S-06), target_id -> chunk
#   summary  - a chunk/document summary (S-07, S-14), target_id -> chunk or parent
#   parent   - a larger parent section / full-document text (S-01, C-135 docstore);
#              never matched by search, fetched by id for context expansion
SEARCHABLE_KINDS = ("chunk", "question", "summary")


@dataclass
class VectorRecord:
    """A single record persisted to the vector index (C-131)."""

    chunk_id: str
    content: str
    embedding: list[float] | None
    source: str
    collection: str
    chunk_index: int = 0
    content_hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    doc_id: str = ""
    title: str = ""
    section_path: str = ""
    acl: list[str] = field(default_factory=lambda: [PUBLIC])
    doc_type: str = ""
    language: str = ""
    tags: list[str] = field(default_factory=list)
    effective_date: str | None = None  # ISO 8601
    page: int | None = None
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    kind: str = "chunk"
    target_id: str = ""
    embedding_model: str = ""
    # Ingest run that wrote the record; older versions of a source are pruned after upsert.
    ingest_version: str = ""


@dataclass
class VectorSearchResult:
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, str] = field(default_factory=dict)
    doc_id: str = ""
    title: str = ""
    section_path: str = ""
    acl: list[str] = field(default_factory=lambda: [PUBLIC])
    effective_date: str | None = None
    page: int | None = None
    chunk_index: int = 0
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    kind: str = "chunk"
    target_id: str = ""
    # Parent expansion: the child chunks that matched, as {chunk_id, score, preview}.
    matched_chunks: list[dict] = field(default_factory=list)


class VectorStore(ABC):
    """Abstraction over the backing vector database so providers are swappable.

    Every read takes RetrievalFilters and must enforce its ACL (C-192); callers never
    filter results themselves.
    """

    async def ensure_index(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Create the index/collection schema if it does not exist."""

    async def ping(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Raise if the store is unreachable (used by the readiness probe)."""

    async def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release network resources on shutdown."""

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> int:
        """Insert or update records by id (idempotent, C-136). Returns the number written."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float] | None,
        filters: RetrievalFilters,
        top_k: int,
        query_text: str | None = None,
        kinds: tuple[str, ...] = SEARCHABLE_KINDS,
        hybrid: bool | None = None,
    ) -> list[VectorSearchResult]:
        """Top-k records. Embedding only = vector; text only = keyword (BM25); both = hybrid.

        `hybrid` overrides RETRIEVAL_MODE for this call when both are given: True fuses the
        keyword side even under RETRIEVAL_MODE=vector, False searches vectors only, None
        follows RETRIEVAL_MODE."""

    @abstractmethod
    async def get_by_ids(
        self, ids: list[str], filters: RetrievalFilters
    ) -> list[VectorSearchResult]:
        """Fetch records by id (docstore lookups, context expansion); ACL enforced."""

    @abstractmethod
    async def get_document_chunks(
        self, doc_id: str, filters: RetrievalFilters, limit: int = 200
    ) -> list[VectorSearchResult]:
        """All chunks of a document in reading order (fetch_document tool, A-14)."""

    @abstractmethod
    async def count(self, collection: str, source: str | None = None) -> int:
        """Record count, for reconciliation with pipeline output (C-139)."""

    @abstractmethod
    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        """Change permissions of a source's records without re-embedding (C-145)."""

    @abstractmethod
    async def delete_by_source(
        self,
        source: str,
        collection: str,
        except_version: str | None = None,
        keep_ids: Collection[str] = (),
    ) -> int:
        """Delete a source's records (all, or all not written by `except_version`).

        `keep_ids` are never deleted: the records the caller has just written. A search index
        that is only eventually consistent (Atlas Search) can still match the previous
        version of a reused id right after the upsert, which would delete the new record."""

    @abstractmethod
    async def delete_collection(self, collection: str) -> int:
        """Remove all records belonging to a collection/namespace. Returns the number deleted."""

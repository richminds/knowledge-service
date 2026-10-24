"""Ingestion manifest (C-148): one row per (collection, source) with status, hashes,
versions and lineage. Drives change detection (C-143), ACL-only updates (C-145),
delete propagation (C-144), near-duplicate detection (C-092), lineage (C-150) and
count reconciliation (C-139).

MANIFEST_STORE=memory (default, per process) | mongo (shared; collection MANIFEST_COLLECTION
in MONGODB_DATABASE, next to the chunks and the graph).
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from app.rag.ingestion.governance import similarity

STATUSES = (
    "pending",
    "parsed",
    "indexed",
    "graphed",
    "skipped",
    "quarantined",
    "failed",
    "deleted",
)


@dataclass
class ManifestEntry:
    collection: str
    source: str
    doc_id: str
    status: str = "pending"
    content_hash: str = ""
    etag: str = ""
    last_modified: str = ""  # source modification time (C-093 canonical version)
    acl: list[str] = field(default_factory=list)
    title: str = ""
    version: str = ""  # ingest version written to the index
    run_id: str = ""  # C-150 lineage: which pipeline run produced it
    pipeline_version: str = ""
    chunk_count: int = 0
    entity_count: int = 0
    duplicate_of: str = ""
    minhash: list[int] = field(default_factory=list)
    error: str = ""
    reasons: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class Manifest(Protocol):
    async def get(self, collection: str, source: str) -> ManifestEntry | None: ...
    async def upsert(self, entry: ManifestEntry) -> None: ...
    async def list_entries(self, collection: str, prefix: str = "") -> list[ManifestEntry]: ...
    async def find_duplicate(
        self, collection: str, source: str, content_hash: str, minhash: list[int], threshold: float
    ) -> ManifestEntry | None: ...


class InMemoryManifest:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], ManifestEntry] = {}

    async def get(self, collection: str, source: str) -> ManifestEntry | None:
        return self._rows.get((collection, source))

    async def upsert(self, entry: ManifestEntry) -> None:
        entry.updated_at = datetime.now(UTC).isoformat()
        self._rows[(entry.collection, entry.source)] = entry

    async def list_entries(self, collection: str, prefix: str = "") -> list[ManifestEntry]:
        return [e for (c, s), e in self._rows.items() if c == collection and s.startswith(prefix)]

    async def find_duplicate(self, collection, source, content_hash, minhash, threshold):
        for entry in await self.list_entries(collection):
            if entry.source == source or entry.status not in ("indexed", "graphed"):
                continue
            if (
                entry.content_hash == content_hash
                or similarity(minhash, entry.minhash) >= threshold
            ):
                return entry
        return None


class MongoManifest:
    """One document per (collection, source); the entry's fields stored as-is."""

    def __init__(self, documents) -> None:
        self._documents = documents  # an AsyncCollection
        self._ready = False

    async def _ensure(self) -> None:
        if not self._ready:
            await self._documents.create_index([("collection", 1), ("source", 1)], unique=True)
            await self._documents.create_index([("collection", 1), ("content_hash", 1)])
            self._ready = True

    @staticmethod
    def _entry(doc: dict) -> ManifestEntry:
        doc.pop("_id", None)
        known = ManifestEntry.__dataclass_fields__
        return ManifestEntry(**{k: v for k, v in doc.items() if k in known})

    async def get(self, collection: str, source: str) -> ManifestEntry | None:
        await self._ensure()
        doc = await self._documents.find_one({"collection": collection, "source": source})
        return self._entry(doc) if doc else None

    async def upsert(self, entry: ManifestEntry) -> None:
        await self._ensure()
        entry.updated_at = datetime.now(UTC).isoformat()
        await self._documents.replace_one(
            {"collection": entry.collection, "source": entry.source}, asdict(entry), upsert=True
        )

    async def list_entries(self, collection: str, prefix: str = "") -> list[ManifestEntry]:
        await self._ensure()
        query: dict = {"collection": collection}
        if prefix:
            query["source"] = {"$regex": "^" + re.escape(prefix)}
        return [self._entry(doc) async for doc in self._documents.find(query)]

    async def find_duplicate(self, collection, source, content_hash, minhash, threshold):
        # Near-duplicates compared in Python (add LSH buckets at scale).
        for entry in await self.list_entries(collection):
            if entry.source == source or entry.status not in ("indexed", "graphed"):
                continue
            if (
                entry.content_hash == content_hash
                or similarity(minhash, entry.minhash) >= threshold
            ):
                return entry
        return None

    async def close(self) -> None:
        """The client is shared (app.core.mongo); nothing to release here."""

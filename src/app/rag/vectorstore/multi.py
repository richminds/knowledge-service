"""Several vector stores behind one VectorStore: write to all, read from the selected one.

Ingestion (upsert, ACL change, delete) is applied to every store so they hold the same
content. Reads go to the store chosen for the current request (`set_vector_backend`), or the
default one - so the semantic strategy, agent tools and graph fallback honour a per-request
choice without knowing about it. This is what lets the two backends be compared on the same
questions.
"""

import asyncio
from collections.abc import Awaitable, Callable, Collection
from contextvars import ContextVar
from typing import TypeVar

from app.core.exceptions import VectorBackendError, VectorStoreError
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import (
    SEARCHABLE_KINDS,
    VectorRecord,
    VectorSearchResult,
    VectorStore,
)

T = TypeVar("T")

_selected: ContextVar[str | None] = ContextVar("vector_backend", default=None)


def set_vector_backend(name: str | None) -> None:
    """Choose the store reads use for the rest of this request (None = the default)."""
    _selected.set(name or None)


def current_vector_backend() -> str | None:
    return _selected.get()


class MultiVectorStore(VectorStore):
    def __init__(self, stores: dict[str, VectorStore], default: str) -> None:
        if default not in stores:
            raise VectorStoreError(f"Default vector store '{default}' is not configured")
        self._stores = stores
        self._default = default

    @property
    def names(self) -> list[str]:
        return list(self._stores)

    @property
    def default(self) -> str:
        return self._default

    @property
    def stores(self) -> dict[str, VectorStore]:
        return dict(self._stores)

    def resolve(self, name: str | None = None) -> str:
        """The store a read uses: the explicit name, else the request's choice, else default."""
        chosen = name or _selected.get() or self._default
        if chosen not in self._stores:
            available = ", ".join(self._stores)
            raise VectorBackendError(f"Vector store '{chosen}' is not enabled ({available})")
        return chosen

    def _reader(self) -> VectorStore:
        return self._stores[self.resolve()]

    async def _all(self, call: Callable[[VectorStore], Awaitable[T]], action: str) -> dict[str, T]:
        """Run `call` on every store; any failure fails the whole write (stores must agree)."""
        names = list(self._stores)
        outcomes = await asyncio.gather(
            *(call(self._stores[n]) for n in names), return_exceptions=True
        )
        failures = [
            f"{n}: {o}" for n, o in zip(names, outcomes, strict=True) if isinstance(o, Exception)
        ]
        if failures:
            raise VectorStoreError(f"{action} failed on {'; '.join(failures)}")
        return dict(zip(names, outcomes, strict=True))  # type: ignore[arg-type]

    # -- lifecycle ----------------------------------------------------------------------
    async def ensure_index(self) -> None:
        await self._all(lambda s: s.ensure_index(), "ensure_index")

    async def ping(self) -> None:
        await self._all(lambda s: s.ping(), "ping")

    async def close(self) -> None:
        await asyncio.gather(*(s.close() for s in self._stores.values()), return_exceptions=True)

    # -- writes: every store ------------------------------------------------------------
    async def upsert(self, records: list[VectorRecord]) -> int:
        written = await self._all(lambda s: s.upsert(records), "upsert")
        return min(written.values())

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        updated = await self._all(lambda s: s.update_acl(source, collection, acl), "update_acl")
        return updated[self._default]

    async def delete_by_source(
        self,
        source: str,
        collection: str,
        except_version: str | None = None,
        keep_ids: Collection[str] = (),
    ) -> int:
        deleted = await self._all(
            lambda s: s.delete_by_source(source, collection, except_version, keep_ids),
            "delete_by_source",
        )
        return deleted[self._default]

    async def delete_collection(self, collection: str) -> int:
        deleted = await self._all(lambda s: s.delete_collection(collection), "delete_collection")
        return deleted[self._default]

    async def count(self, collection: str, source: str | None = None) -> int:
        # The smallest count: a store that fell behind shows up in reconciliation (C-139).
        counts = await self._all(lambda s: s.count(collection, source), "count")
        return min(counts.values())

    # -- reads: the selected store ------------------------------------------------------
    async def search(
        self,
        query_embedding: list[float] | None,
        filters: RetrievalFilters,
        top_k: int,
        query_text: str | None = None,
        kinds: tuple[str, ...] = SEARCHABLE_KINDS,
        hybrid: bool | None = None,
    ) -> list[VectorSearchResult]:
        return await self._reader().search(
            query_embedding, filters, top_k, query_text, kinds, hybrid=hybrid
        )

    async def get_by_ids(
        self, ids: list[str], filters: RetrievalFilters
    ) -> list[VectorSearchResult]:
        return await self._reader().get_by_ids(ids, filters)

    async def get_document_chunks(
        self, doc_id: str, filters: RetrievalFilters, limit: int = 200
    ) -> list[VectorSearchResult]:
        return await self._reader().get_document_chunks(doc_id, filters, limit)

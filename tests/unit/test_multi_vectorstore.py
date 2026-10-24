"""MultiVectorStore: writes reach every store, reads follow the request's selected store."""

import pytest

from app.core.exceptions import VectorBackendError, VectorStoreError
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorRecord
from app.rag.vectorstore.multi import MultiVectorStore, set_vector_backend
from tests.fakes import FakeVectorStore

FILTERS = RetrievalFilters(collection="c1")


def _record(chunk_id: str, content: str, embedding: list[float]) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id, content=content, embedding=embedding, source="doc.md", collection="c1"
    )


@pytest.fixture
def stores() -> dict[str, FakeVectorStore]:
    return {"mongo": FakeVectorStore(), "mongo_b": FakeVectorStore()}


@pytest.fixture
def multi(stores) -> MultiVectorStore:
    set_vector_backend(None)
    return MultiVectorStore(dict(stores), default="mongo")


async def test_upsert_writes_to_every_store(multi, stores):
    assert await multi.upsert([_record("a", "leave policy", [1.0, 0.0])]) == 1

    assert all("a" in s.records for s in stores.values())


async def test_reads_use_default_then_the_selected_store(multi, stores):
    await stores["mongo"].upsert([_record("only-a", "leave", [1.0, 0.0])])
    await stores["mongo_b"].upsert([_record("only-b", "leave", [1.0, 0.0])])

    default = await multi.search([1.0, 0.0], FILTERS, 5, query_text="leave")
    assert [r.chunk_id for r in default] == ["only-a"]

    set_vector_backend("mongo_b")
    selected = await multi.search([1.0, 0.0], FILTERS, 5, query_text="leave")
    assert [r.chunk_id for r in selected] == ["only-b"]
    assert [r.chunk_id for r in await multi.get_by_ids(["only-b"], FILTERS)] == ["only-b"]
    set_vector_backend(None)


async def test_unknown_backend_is_rejected(multi):
    set_vector_backend("qdrant")
    try:
        with pytest.raises(VectorBackendError):
            await multi.search([1.0, 0.0], FILTERS, 5)
    finally:
        set_vector_backend(None)


async def test_a_failing_store_fails_the_write_and_names_it(multi, stores, monkeypatch):
    async def boom(records):
        raise VectorStoreError("down")

    monkeypatch.setattr(stores["mongo_b"], "upsert", boom)

    with pytest.raises(VectorStoreError, match="mongo_b"):
        await multi.upsert([_record("a", "x", [1.0])])


async def test_count_reports_the_laggard_so_reconciliation_flags_it(multi, stores):
    await stores["mongo"].upsert([_record("a", "x", [1.0]), _record("b", "y", [1.0])])
    await stores["mongo_b"].upsert([_record("a", "x", [1.0])])

    assert await multi.count("c1") == 1


async def test_deletes_apply_to_every_store(multi, stores):
    await multi.upsert([_record("a", "x", [1.0])])

    await multi.delete_collection("c1")

    assert all(not s.records for s in stores.values())


def test_default_must_be_a_configured_store(stores):
    with pytest.raises(VectorStoreError):
        MultiVectorStore(stores, default="qdrant")


async def test_prune_spares_records_just_written_even_if_index_is_stale():
    """Atlas Search is near-real-time: right after an upsert, the prune query can still see
    the previous version of a reused id. keep_ids must stop it deleting the new record."""
    from app.rag.vectorstore.base import VectorRecord
    from tests.fakes import FakeVectorStore

    store = FakeVectorStore()
    stale = VectorRecord("doc-0000", "old", [0.1], "a.md", "c", ingest_version="v1")
    fresh_only = VectorRecord("doc-0001", "old", [0.1], "a.md", "c", ingest_version="v1")
    await store.upsert([stale, fresh_only])
    # The new version rewrote doc-0000, but the index still reports v1 for it.
    deleted = await store.delete_by_source("a.md", "c", except_version="v2", keep_ids={"doc-0000"})

    assert deleted == 1
    assert set(store.records) == {"doc-0000"}

"""Integration test for the ingestion pipeline (``rag.ingestion.ingest()``).

Exercises the full LangGraph pipeline — parse_and_load, chunk,
embed_and_insert, graph_insert — wired together for real: file parsing,
chunking, and graph orchestration all run unmocked. Only the true I/O
boundaries are replaced with in-memory fakes (the embedding HTTP call to the
LLM Gateway, and MongoDB writes), the same "mock only the provider boundary"
approach the LLM Gateway's own test suite uses.
"""
from __future__ import annotations

import pytest

from rag.ingestion import ingest
from rag.parent_store import MongoParentStore
from rag.vector_store import VectorStore


class FakeEmbeddingsClient:
    """Stand-in for GatewayEmbeddingsClient — deterministic, no HTTP."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 0.1, 0.2] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 0.9, 0.8]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@pytest.fixture()
def fake_ingestion_backends(monkeypatch):
    """Replace the Mongo/embedding I/O boundaries with in-memory fakes."""
    embeddings = FakeEmbeddingsClient()
    monkeypatch.setattr("rag.ingestion.get_embeddings", lambda: embeddings)

    saved_parents: dict = {}

    async def fake_save(self, parent_documents):
        saved_parents.update(parent_documents)

    monkeypatch.setattr(MongoParentStore, "save", fake_save)

    upserted_chunks: list = []

    async def fake_upsert(self, chunks):
        upserted_chunks.extend(chunks)
        return len(chunks)

    monkeypatch.setattr(VectorStore, "upsert", fake_upsert)

    return {"parents": saved_parents, "chunks": upserted_chunks}


async def test_ingest_pipeline_end_to_end(tmp_path, fake_ingestion_backends):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Refund Policy\n\n"
        "Refunds are issued within five business days of a return request.\n\n"
        "## Exceptions\n\n"
        "Final-sale items are not eligible for a refund under any circumstances.\n",
        encoding="utf-8",
    )

    result = await ingest(input_paths=[str(doc_path)], chunk_strategy="recursive")

    assert result["inserted_count"] > 0
    assert result["inserted_count"] == len(fake_ingestion_backends["chunks"])
    assert fake_ingestion_backends["parents"]  # parent context groups were persisted
    assert result["graph_inserted_count"] == 0  # graph retrieval disabled by default

    # Every persisted chunk carries the metadata the retrieval pipeline depends on.
    for chunk in fake_ingestion_backends["chunks"]:
        assert chunk.metadata["source"] == str(doc_path)
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["parent_id"] in fake_ingestion_backends["parents"]


async def test_ingest_pipeline_semantic_strategy_uses_embeddings(
    tmp_path, fake_ingestion_backends
):
    doc_path = tmp_path / "notes.txt"
    doc_path.write_text(
        "Alpha topic covers apples in detail across several sentences of content.\n\n"
        "Beta topic is unrelated and discusses spacecraft engines and thrust vectors.\n\n"
        "Gamma topic returns to apples and fruit again for good measure.\n",
        encoding="utf-8",
    )

    result = await ingest(input_paths=[str(doc_path)], chunk_strategy="semantic")

    assert result["inserted_count"] > 0
    assert all(
        c.metadata["chunk_strategy"] == "semantic" for c in fake_ingestion_backends["chunks"]
    )


async def test_ingest_pipeline_empty_directory_produces_no_chunks(
    tmp_path, fake_ingestion_backends
):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = await ingest(input_paths=[str(empty_dir)])

    assert result["inserted_count"] == 0
    assert fake_ingestion_backends["chunks"] == []


async def test_ingest_pipeline_records_uploaded_by_on_chunks_and_parents(
    tmp_path, fake_ingestion_backends
):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Refund Policy\n\nRefunds are issued within five business days.\n",
        encoding="utf-8",
    )

    await ingest(
        input_paths=[str(doc_path)], chunk_strategy="recursive", uploaded_by="alice@example.com"
    )

    assert fake_ingestion_backends["chunks"]
    assert all(
        c.metadata["uploaded_by"] == "alice@example.com"
        for c in fake_ingestion_backends["chunks"]
    )
    assert fake_ingestion_backends["parents"]
    assert all(
        p.metadata.get("uploaded_by") == "alice@example.com"
        for p in fake_ingestion_backends["parents"].values()
    )
    # Access control stays public by default — uploaded_by is provenance only.
    assert all(
        c.metadata["authorized_users"] == "*" for c in fake_ingestion_backends["chunks"]
    )


async def test_ingest_pipeline_defaults_uploaded_by_to_empty_string(
    tmp_path, fake_ingestion_backends
):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text("# Refund Policy\n\nRefunds take five days.\n", encoding="utf-8")

    await ingest(input_paths=[str(doc_path)])

    assert fake_ingestion_backends["chunks"]
    assert all(c.metadata["uploaded_by"] == "" for c in fake_ingestion_backends["chunks"])


async def test_ingest_pipeline_records_org_id_on_chunks(tmp_path, fake_ingestion_backends):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Refund Policy\n\nRefunds are issued within five business days.\n",
        encoding="utf-8",
    )

    await ingest(input_paths=[str(doc_path)], chunk_strategy="recursive", org_id="org-a")

    assert fake_ingestion_backends["chunks"]
    assert all(c.metadata["org_id"] == "org-a" for c in fake_ingestion_backends["chunks"])


async def test_ingest_pipeline_defaults_org_id_to_unscoped(tmp_path, fake_ingestion_backends):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text("# Refund Policy\n\nRefunds take five days.\n", encoding="utf-8")

    await ingest(input_paths=[str(doc_path)])

    assert fake_ingestion_backends["chunks"]
    assert all(c.metadata["org_id"] == "*" for c in fake_ingestion_backends["chunks"])

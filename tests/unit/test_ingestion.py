import pytest

from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.connectors.local import resolve_local_path
from app.rag.ingestion.models import SourceDocument
from app.rag.ingestion.parsing import ParserRouter
from app.rag.ingestion.pipeline import LoadedDocument, stable_chunk_id


async def test_pipeline_ingests_supported_files_and_skips_others(ingestion_pipeline, vector_store):
    stats = await ingestion_pipeline.run("docs", "default", {"owner": "hr"})

    assert stats.document_count == 2
    assert stats.chunk_count == len(vector_store.records) > 0
    record = next(iter(vector_store.records.values()))
    assert record.metadata["owner"] == "hr"
    assert record.metadata["pipeline_version"]
    assert "ignored.bin" not in {r.source.rsplit("/", 1)[-1] for r in vector_store.records.values()}


async def test_reingesting_shorter_document_prunes_stale_chunks(ingestion_pipeline, vector_store):
    long_doc = LoadedDocument(source="a.md", text="word " * 400)
    await ingestion_pipeline.run_for_documents([long_doc], "default")
    long_count = len(vector_store.records)

    short_doc = LoadedDocument(source="a.md", text="word " * 30)
    await ingestion_pipeline.run_for_documents([short_doc], "default")

    assert long_count > 1
    assert len(vector_store.records) == 1


async def test_same_source_in_two_collections_does_not_collide(ingestion_pipeline, vector_store):
    doc = LoadedDocument(source="a.md", text="shared content")
    await ingestion_pipeline.run_for_documents([doc], "alpha")
    await ingestion_pipeline.run_for_documents([doc], "beta")

    assert {r.collection for r in vector_store.records.values()} == {"alpha", "beta"}
    assert stable_chunk_id("alpha", "a.md", 0) != stable_chunk_id("beta", "a.md", 0)


async def test_local_path_outside_root_is_rejected(ingestion_pipeline):
    with pytest.raises(DocumentLoadError, match="inside the ingestion root"):
        await ingestion_pipeline.run("../../etc", "default")


def test_resolve_local_path_allows_paths_inside_root(data_root):
    assert resolve_local_path("docs", data_root) == (data_root / "docs").resolve()


async def test_parser_rejects_executables_by_content():
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        await ParserRouter().parse(
            SourceDocument(source="x", name="report.pdf", content=b"MZ\x90\x00")
        )


async def test_upload_uses_basename_only(ingestion_pipeline, vector_store):
    await ingestion_pipeline.run_for_upload("../../secret/notes.txt", b"hello world", "default")

    assert {r.source for r in vector_store.records.values()} == {"upload://notes.txt"}

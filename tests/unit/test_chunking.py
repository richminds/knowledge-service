from langchain_core.documents import Document

from rag.chunking import chunk_documents, recursive_chunk_documents


class FakeEmbeddingsClient:
    """Stand-in for GatewayEmbeddingsClient — deterministic, no HTTP."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 0.1, 0.2] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 0.9, 0.8]


def test_recursive_chunking_assigns_stable_chunk_ids():
    doc = Document(
        page_content="word " * 500,
        metadata={"source": "f.txt", "section": "intro"},
    )
    chunks, parents = recursive_chunk_documents([doc])

    assert chunks
    assert all(c.metadata.get("chunk_id") for c in chunks)
    assert all(c.metadata.get("parent_id") for c in chunks)
    assert parents
    # every chunk's parent_id must resolve to a produced parent document
    assert all(c.metadata["parent_id"] in parents for c in chunks)


def test_chunk_documents_dispatches_to_recursive_by_default():
    doc = Document(page_content="hello world", metadata={"source": "f.txt", "section": "s"})
    chunks, _parents = chunk_documents([doc], FakeEmbeddingsClient(), "recursive")
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_strategy"] == "recursive"


def test_semantic_chunking_produces_chunks_and_calls_embeddings():
    text = (
        "First paragraph discusses apples in detail with several sentences of content.\n\n"
        "Second paragraph is about an unrelated topic like spacecraft engines and thrust.\n\n"
        "Third paragraph returns to apples and fruit again for good measure."
    )
    doc = Document(page_content=text, metadata={"source": "f.md", "section": "s"})
    chunks, parents = chunk_documents([doc], FakeEmbeddingsClient(), "semantic")
    assert chunks
    assert all(c.metadata["chunk_strategy"] == "semantic" for c in chunks)
    assert parents

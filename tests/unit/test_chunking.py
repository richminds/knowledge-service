from app.rag.ingestion.chunking import chunk_text


def test_chunk_text_splits_on_token_boundaries():
    text = "word " * 2000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=20) == []


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    try:
        chunk_text("hello world", chunk_size=50, chunk_overlap=50)
        assert False, "expected ValueError"
    except ValueError:
        pass

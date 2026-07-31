from pathlib import Path

from rag.loader import format_document, load_documents


def test_format_document_records_uploaded_by(tmp_path):
    path = tmp_path / "policy.md"
    doc = format_document(path, "# Policy\n\nRefunds take five days.", uploaded_by="alice")

    assert doc.metadata["uploaded_by"] == "alice"
    # Provenance is separate from the access-control list, which stays public
    # by default regardless of who uploaded the document.
    assert doc.metadata["authorized_users"] == "*"
    assert doc.metadata["authorized_teams"] == "*"


def test_format_document_defaults_uploaded_by_to_empty_string(tmp_path):
    path = tmp_path / "policy.md"
    doc = format_document(path, "# Policy\n\nRefunds take five days.")

    assert doc.metadata["uploaded_by"] == ""


def test_load_documents_threads_uploaded_by_through_markdown_sections(tmp_path):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Refund Policy\n\nRefunds take five days.\n\n## Exceptions\n\nFinal sale excluded.\n",
        encoding="utf-8",
    )

    docs = load_documents([Path(doc_path)], uploaded_by="bob@example.com")

    assert len(docs) >= 2  # split into sections
    assert all(d.metadata["uploaded_by"] == "bob@example.com" for d in docs)


def test_load_documents_threads_uploaded_by_through_txt(tmp_path):
    doc_path = tmp_path / "notes.txt"
    doc_path.write_text("Plain text notes about the refund policy.", encoding="utf-8")

    docs = load_documents([Path(doc_path)], uploaded_by="carol")

    assert len(docs) == 1
    assert docs[0].metadata["uploaded_by"] == "carol"

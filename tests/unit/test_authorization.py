from langchain_core.documents import Document

from rag.authorization import filter_authorized_results, is_authorized_document
from rag.vector_store import ScoredDocument


def _doc(**metadata) -> Document:
    return Document(page_content="x", metadata=metadata)


def test_public_document_is_always_authorized():
    doc = _doc(authorized_users="*", authorized_teams="*")
    assert is_authorized_document(doc) is True
    assert is_authorized_document(doc, user_id="anyone") is True


def test_user_specific_authorization():
    doc = _doc(authorized_users="alice,bob", authorized_teams="")
    assert is_authorized_document(doc, user_id="alice") is True
    assert is_authorized_document(doc, user_id="carol") is False


def test_team_specific_authorization():
    doc = _doc(authorized_users="", authorized_teams="team-a")
    assert is_authorized_document(doc, team_id="team-a") is True
    assert is_authorized_document(doc, team_id="team-b") is False


def test_default_metadata_is_public():
    # authorized_users/authorized_teams default to "*" when absent (loader.py)
    doc = _doc()
    assert is_authorized_document(doc, user_id="anyone") is True


def test_filter_authorized_results_drops_private_docs():
    public = ScoredDocument(
        document=_doc(authorized_users="*", authorized_teams="*"), score=1.0, source="semantic"
    )
    private = ScoredDocument(
        document=_doc(authorized_users="alice", authorized_teams=""), score=1.0, source="semantic"
    )
    results = filter_authorized_results([public, private], user_id="bob")
    assert results == [public]

    results_alice = filter_authorized_results([public, private], user_id="alice")
    assert results_alice == [public, private]

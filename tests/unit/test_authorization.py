from langchain_core.documents import Document

from rag.authorization import filter_authorized_results, is_authorized_document
from rag.vector_store import ScoredDocument

# NOTE: the organization gate was removed platform-wide — auth-service no
# longer issues org_id, so nothing could populate it. An ACCOUNT is the only
# scope now, and the account tests below are what enforce isolation.


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


# ─────────────────────────────────────────────── org_id tenant isolation


def test_unscoped_document_is_visible_to_any_org():
    doc = _doc()  # no org_id key at all — same as loader.py's "*" default
    assert is_authorized_document(doc, org_id="org-a") is True
    assert is_authorized_document(doc, org_id=None) is True






def test_unscoped_document_is_visible_to_any_account():
    doc = _doc()  # no account_id key — same as loader.py's "*" default
    assert is_authorized_document(doc, account_id="ingest") is True
    assert is_authorized_document(doc, account_id=None) is True


def test_account_scoped_document_is_only_visible_under_its_own_account():
    doc = _doc(account_id="ingest", authorized_users="*", authorized_teams="*")
    assert is_authorized_document(doc, account_id="ingest") is True
    assert is_authorized_document(doc, account_id="portal") is False
    # Fail closed for a caller that names no account at all.
    assert is_authorized_document(doc) is False



def test_filter_authorized_results_enforces_account_isolation():
    ingest_doc = ScoredDocument(document=_doc(account_id="ingest"), score=1.0, source="semantic")
    portal_doc = ScoredDocument(document=_doc(account_id="portal"), score=1.0, source="semantic")
    public_doc = ScoredDocument(document=_doc(), score=1.0, source="semantic")

    results = filter_authorized_results(
        [ingest_doc, portal_doc, public_doc], account_id="ingest"
    )
    assert results == [ingest_doc, public_doc]

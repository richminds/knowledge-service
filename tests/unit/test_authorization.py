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


# ─────────────────────────────────────────────── org_id tenant isolation


def test_unscoped_document_is_visible_to_any_org():
    doc = _doc()  # no org_id key at all — same as loader.py's "*" default
    assert is_authorized_document(doc, org_id="org-a") is True
    assert is_authorized_document(doc, org_id=None) is True


def test_org_scoped_document_is_only_visible_to_its_own_org():
    doc = _doc(org_id="org-a", authorized_users="*", authorized_teams="*")
    assert is_authorized_document(doc, org_id="org-a") is True
    assert is_authorized_document(doc, org_id="org-b") is False


def test_org_scoped_document_is_hidden_when_caller_gives_no_org_id():
    # Fail closed: a caller that doesn't identify its org must not see
    # another org's private data just because it omitted org_id.
    doc = _doc(org_id="org-a", authorized_users="*", authorized_teams="*")
    assert is_authorized_document(doc) is False
    assert is_authorized_document(doc, org_id="") is False


def test_org_gate_is_a_hard_and_not_folded_into_the_user_team_or():
    # Matching user_id/team_id must NOT bypass a mismatched org_id — org is a
    # strict tenant boundary, unlike the permissive user/team OR-list.
    doc = _doc(org_id="org-a", authorized_users="alice", authorized_teams="team-a")
    assert is_authorized_document(doc, user_id="alice", org_id="org-b") is False
    assert is_authorized_document(doc, team_id="team-a", org_id="org-b") is False
    assert is_authorized_document(doc, user_id="alice", org_id="org-a") is True


def test_filter_authorized_results_enforces_org_isolation():
    org_a_doc = ScoredDocument(document=_doc(org_id="org-a"), score=1.0, source="semantic")
    org_b_doc = ScoredDocument(document=_doc(org_id="org-b"), score=1.0, source="semantic")
    public_doc = ScoredDocument(document=_doc(), score=1.0, source="semantic")

    results = filter_authorized_results([org_a_doc, org_b_doc, public_doc], org_id="org-a")
    assert results == [org_a_doc, public_doc]


# ─────────────────────────────────────────── account_id application isolation


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


def test_account_gate_is_independent_of_the_org_gate():
    # Both are hard ANDs: matching one doesn't excuse missing the other.
    doc = _doc(account_id="ingest", org_id="org-a")
    assert is_authorized_document(doc, account_id="ingest", org_id="org-a") is True
    assert is_authorized_document(doc, account_id="ingest", org_id="org-b") is False
    assert is_authorized_document(doc, account_id="portal", org_id="org-a") is False


def test_filter_authorized_results_enforces_account_isolation():
    ingest_doc = ScoredDocument(document=_doc(account_id="ingest"), score=1.0, source="semantic")
    portal_doc = ScoredDocument(document=_doc(account_id="portal"), score=1.0, source="semantic")
    public_doc = ScoredDocument(document=_doc(), score=1.0, source="semantic")

    results = filter_authorized_results(
        [ingest_doc, portal_doc, public_doc], account_id="ingest"
    )
    assert results == [ingest_doc, public_doc]

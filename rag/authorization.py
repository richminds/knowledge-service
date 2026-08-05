from __future__ import annotations

from langchain_core.documents import Document

from .vector_store import ScoredDocument


def filter_authorized_results(
    results: list[ScoredDocument],
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
) -> list[ScoredDocument]:
    """Filter retrieved documents by org, user, and team authorization metadata.

    Implementation:
        The function first enforces the `org_id` tenant boundary (a document
        tagged with a real org_id is only visible to a query from that same
        org_id; untagged/`*` documents are visible to everyone) — see
        `is_authorized_document` for why this is a hard gate rather than
        folded into the user/team check below it. Within that boundary, it
        keeps public documents marked with `*`, documents that list the
        requested `user_id` in `authorized_users`, or documents that list the
        requested `team_id` in `authorized_teams`. Metadata values can be a
        comma separated string or a list.

    Usage:
        The query graph applies this after semantic and keyword retrieval.

    How it helps other functions:
        It prevents unauthorized chunks from entering hybrid fusion, reranking,
        prompt augmentation, generation, citation validation, and evaluation.
    """
    return [
        result
        for result in results
        if is_authorized_document(
            result.document, user_id=user_id, team_id=team_id, org_id=org_id
        )
    ]


def is_authorized_document(
    document: Document,
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
) -> bool:
    """Check whether one document is visible to a user, team, and org.

    Implementation:
        `org_id` is a hard AND gate, checked first and independently of
        everything else: a document tagged with a specific org_id (not `*`)
        is only visible to a caller supplying that same org_id, full stop —
        this is what makes org_id an actual tenant-isolation boundary rather
        than another entry in the permissive user/team OR-list below it (an
        OR would let any matching user_id/team_id see across organizations,
        defeating the isolation). Once past that gate, the function reads
        `authorized_users` and `authorized_teams` from metadata, normalizes
        them into sets, and treats `*` as public access within the org.

    Usage:
        `filter_authorized_results` calls this for every retrieved result.

    How it helps other functions:
        Centralizing the rule makes it easy to replace this with enterprise
        RBAC/ABAC checks later, or to have a calling application own its own
        authorization model on top of this default.
    """
    doc_org = str(document.metadata.get("org_id") or "*")
    if doc_org != "*" and doc_org != org_id:
        return False

    users = _metadata_values(document.metadata.get("authorized_users", "*"))
    teams = _metadata_values(document.metadata.get("authorized_teams", "*"))
    return (
        "*" in users
        or "*" in teams
        or (bool(user_id) and user_id in users)
        or (bool(team_id) and team_id in teams)
    )


def _metadata_values(value: object) -> set[str]:
    """Normalize authorization metadata into a set of strings.

    Implementation:
        Strings are split on commas, lists/tuples/sets are converted item by
        item, and empty values produce an empty set.

    Usage:
        `is_authorized_document` uses this for both user and team metadata.

    How it helps other functions:
        It allows loaders or future connectors to provide authorization metadata
        in a few simple shapes without changing retrieval filtering logic.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()}

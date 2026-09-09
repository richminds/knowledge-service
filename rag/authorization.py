from __future__ import annotations

from langchain_core.documents import Document

from .vector_store import ScoredDocument


def filter_authorized_results(
    results: list[ScoredDocument],
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> list[ScoredDocument]:
    """Filter retrieved documents by account, org, user, and team metadata.

    Implementation:
        The function first enforces the `account_id` and `org_id` boundaries
        (a document tagged with a real value is only visible to a query
        carrying the same one; untagged/`*` documents are visible to
        everyone) — see `is_authorized_document` for why these are hard gates
        rather than folded into the user/team check below them. Within those
        boundaries, it keeps public documents marked with `*`, documents that
        list the requested `user_id` in `authorized_users`, or documents that
        list the requested `team_id` in `authorized_teams`. Metadata values
        can be a comma separated string or a list.

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
            result.document,
            user_id=user_id,
            team_id=team_id,
            org_id=org_id,
            account_id=account_id,
        )
    ]


def is_authorized_document(
    document: Document,
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> bool:
    """Check whether one document is visible to a user, team, org, and account.

    Implementation:
        `account_id` (the application the caller signed into) and `org_id`
        (their tenant) are hard AND gates, checked first and independently of
        everything else: a document tagged with a specific value (not `*`) is
        only visible to a caller supplying that same value, full stop — this
        is what makes them actual isolation boundaries rather than another
        entry in the permissive user/team OR-list below them (an OR would let
        any matching user_id/team_id see across accounts/organizations,
        defeating the isolation). Once past those gates, the function reads
        `authorized_users` and `authorized_teams` from metadata, normalizes
        them into sets, and treats `*` as public access within the boundary.

    Usage:
        `filter_authorized_results` calls this for every retrieved result.

    How it helps other functions:
        Centralizing the rule makes it easy to replace this with enterprise
        RBAC/ABAC checks later, or to have a calling application own its own
        authorization model on top of this default.
    """
    doc_account = str(document.metadata.get("account_id") or "*")
    if doc_account != "*" and doc_account != account_id:
        return False

    # The organization boundary was removed: an ACCOUNT is now the only scope
    # (auth-service no longer issues org_id, so nothing could populate it).
    # Documents ingested before that still carry metadata["org_id"]; it is
    # deliberately not read, so they remain visible to their account rather
    # than disappearing behind a filter nothing can satisfy.

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

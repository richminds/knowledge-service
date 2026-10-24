"""Shared retrieval types used by every retriever, strategy and tool (C-022, C-023).

ACL model (C-012, C-066, C-106, C-192): every chunk / graph node carries `acl`, a list of
principal ids allowed to read it (auth-service user ids, account ids, or "*" for
everyone in the organisation). A caller's principals are {"*", user id, account id}.
`RetrievalFilters.principals = None` means access control is not enforced (auth disabled,
local development only).

Account isolation: every chunk / graph node also carries `account_id`, the app account it
was ingested from. `RetrievalFilters.account_id` is the caller's account; retrievers only
return items of that account ("" = the caller has none, so only items ingested without
one). None = not enforced (AUTH_MODE=none, ACCOUNT_ISOLATION=false). It is ANDed with the
ACL: an item must be in the caller's account and readable by the caller.
"""

from dataclasses import dataclass, field, replace
from typing import Iterable

PUBLIC = "*"


def is_allowed(acl: Iterable[str], principals: frozenset[str] | None) -> bool:
    """True when the caller may read an item with this ACL. Empty ACL = nobody (fail closed)."""
    if principals is None:
        return True
    return any(p in principals for p in acl)


@dataclass(frozen=True)
class RetrievalFilters:
    """Everything that narrows a search, applied inside every retriever (never by the LLM)."""

    collection: str = "default"
    principals: frozenset[str] | None = None
    account_id: str | None = None  # see the module docstring
    doc_types: frozenset[str] = frozenset()
    languages: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    effective_from: str | None = None  # ISO date, inclusive
    effective_to: str | None = None  # ISO date, inclusive
    tags: frozenset[str] = field(default_factory=frozenset)
    doc_ids: frozenset[str] = field(default_factory=frozenset)  # hierarchical retrieval (S-26)

    def allows(self, acl: Iterable[str]) -> bool:
        return is_allowed(acl, self.principals)

    def allows_account(self, account_id: str) -> bool:
        return self.account_id is None or (account_id or "") == self.account_id

    def with_changes(self, **changes) -> "RetrievalFilters":
        return replace(self, **changes)

    def relaxed(self) -> "RetrievalFilters":
        """Drop the metadata narrowing (keep collection, ACL and account) — empty-result
        fallback."""
        return RetrievalFilters(
            collection=self.collection, principals=self.principals, account_id=self.account_id
        )

    @property
    def has_metadata_filters(self) -> bool:
        return bool(
            self.doc_types
            or self.languages
            or self.sources
            or self.effective_from
            or self.effective_to
            or self.tags
            or self.doc_ids
        )

    def signature(self) -> str:
        """Stable string for cache keys: two filters with the same signature see the same data."""
        principals = "|".join(sorted(self.principals)) if self.principals is not None else "~"
        account = self.account_id if self.account_id is not None else "~"
        parts = [
            self.collection,
            principals,
            account,
            ",".join(sorted(self.doc_types)),
            ",".join(sorted(self.languages)),
            ",".join(sorted(self.sources)),
            self.effective_from or "",
            self.effective_to or "",
            ",".join(sorted(self.tags)),
            ",".join(sorted(self.doc_ids)),
        ]
        return "\x1f".join(parts)

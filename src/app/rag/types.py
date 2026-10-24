"""Shared retrieval types used by every retriever, strategy and tool (C-022, C-023).

ACL model (C-012, C-066, C-106, C-192): every chunk / graph node carries `acl`, a list of
principal ids allowed to read it (Entra user object ids, group object ids, or "*" for
everyone in the organisation). A caller's principals are {"*", user id, group ids}.
`RetrievalFilters.principals = None` means access control is not enforced (auth disabled,
local development only).
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
    doc_types: frozenset[str] = frozenset()
    languages: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    effective_from: str | None = None  # ISO date, inclusive
    effective_to: str | None = None  # ISO date, inclusive
    tags: frozenset[str] = field(default_factory=frozenset)
    doc_ids: frozenset[str] = field(default_factory=frozenset)  # hierarchical retrieval (S-26)

    def allows(self, acl: Iterable[str]) -> bool:
        return is_allowed(acl, self.principals)

    def with_changes(self, **changes) -> "RetrievalFilters":
        return replace(self, **changes)

    def relaxed(self) -> "RetrievalFilters":
        """Drop the metadata narrowing (keep collection and ACL) — empty-result fallback."""
        return RetrievalFilters(collection=self.collection, principals=self.principals)

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
        parts = [
            self.collection,
            principals,
            ",".join(sorted(self.doc_types)),
            ",".join(sorted(self.languages)),
            ",".join(sorted(self.sources)),
            self.effective_from or "",
            self.effective_to or "",
            ",".join(sorted(self.tags)),
            ",".join(sorted(self.doc_ids)),
        ]
        return "\x1f".join(parts)

"""Document metadata (A7): system (C-101), title/sections (C-102), business rules (C-103),
effective date, document type (C-105), ACL (C-106), pipeline versions (C-107), validated
by the DocumentMetadata schema (C-108).

Business rules live in config/metadata_rules.yaml (placeholder): source-path regexes that
set department, region, doc_type and tags.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from app.rag.ingestion.models import DocumentMetadata, ParsedDocument, SourceDocument

PIPELINE_VERSION = "2.0"
CHUNKER_VERSION = "structure-v1"

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_EFFECTIVE = re.compile(
    rf"\b(?:effective|valid from|in force from|approved on|issued on)\s*(?:date)?\s*[:\-]?\s*"
    rf"(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}"
    rf"|(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}})",
    re.IGNORECASE,
)


@dataclass
class MetadataRule:
    pattern: re.Pattern[str]
    values: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_rules(path: str) -> tuple[MetadataRule, ...]:
    file = Path(path)
    if not file.is_file():
        return ()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return tuple(
        MetadataRule(
            pattern=re.compile(rule["match"], re.IGNORECASE),
            values={k: str(v) for k, v in rule.items() if k not in ("match", "tags")},
            tags=[str(t) for t in rule.get("tags") or []],
        )
        for rule in data.get("rules") or []
    )


def parse_date(text: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d %Y", "%B %d, %Y"):
        try:
            return (
                datetime.strptime(text.strip().replace(",", ", ").replace(",  ", ", "), fmt)
                .date()
                .isoformat()
            )
        except ValueError:
            continue
    return None


def effective_date_from_text(text: str) -> str | None:
    match = _EFFECTIVE.search(text[:5000])
    return parse_date(match.group(1)) if match else None


def build_metadata(
    document: SourceDocument,
    parsed: ParsedDocument,
    *,
    doc_id: str,
    collection: str,
    acl: list[str],
    sensitivity: str,
    rules: tuple[MetadataRule, ...] = (),
    embedding_model: str = "",
    extra: dict[str, str] | None = None,
) -> DocumentMetadata:
    values: dict[str, str] = {}
    tags: list[str] = []
    for rule in rules:  # C-103: later rules override earlier ones
        if rule.pattern.search(document.source):
            values.update(rule.values)
            tags.extend(rule.tags)
    source_meta = {k: v for k, v in document.metadata.items() if isinstance(v, str)}
    return DocumentMetadata(
        doc_id=doc_id,
        source=document.source,
        collection=collection,
        title=parsed.title or document.name,
        url=document.url,
        author=document.author,
        modified=document.last_modified,
        effective_date=(
            source_meta.get("effective_date")
            or effective_date_from_text(parsed.text)
            or (document.last_modified[:10] if document.last_modified else None)
        ),
        doc_type=values.pop("doc_type", source_meta.get("doc_type", "other")),
        language=parsed.language,
        acl=acl,
        tags=sorted(set(tags)),
        department=values.pop("department", ""),
        region=values.pop("region", ""),
        sensitivity=sensitivity,
        version=document.etag,
        content_hash=document.content_hash,
        size_bytes=len(document.content),
        page_count=parsed.page_count,
        pipeline_versions={
            "pipeline": PIPELINE_VERSION,
            "parser": parsed.parser,
            "chunker": CHUNKER_VERSION,
            "embedding": embedding_model,
        },
        extra={**{k: v for k, v in values.items()}, **(extra or {})},
    )

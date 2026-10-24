"""Ingestion data models (C-022, C-086, C-100, C-108).

SourceDocument  - raw bytes + source metadata from a connector (A4)
Element         - one structural unit from parsing: heading, paragraph, table... (A5)
ParsedDocument  - canonical intermediate format: ordered elements + section tree (C-086)
DocumentMetadata- validated metadata schema carried to every chunk (A7)
Chunk           - a retrievable unit with provenance and links (A8)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

ElementType = Literal[
    "title", "heading", "paragraph", "list_item", "table", "code", "figure", "slide", "transcript"
]  # fmt: skip


@dataclass
class SourceDocument:
    """What a connector returns: the original file plus everything known about its source."""

    source: str  # stable URI: path, blob://, sharepoint://, confluence://, https://...
    name: str  # file name, used for type detection fallback and titles
    content: bytes
    content_type: str = ""  # MIME type if the source reports one
    acl: list[str] = field(default_factory=list)  # C-066: permissions travel with the doc
    etag: str = ""  # C-067: change marker
    last_modified: str | None = None  # ISO 8601
    url: str = ""  # human-viewable link for citations (C-188)
    author: str = ""
    metadata: dict[str, str] = field(default_factory=dict)  # source-specific extras
    sensitivity_label: str = ""  # e.g. Microsoft Purview label from the source

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass
class Element:
    type: ElementType
    text: str
    level: int = 0  # heading level (1 = top)
    page: int | None = None  # C-085 provenance
    section_path: list[str] = field(default_factory=list)  # C-084 / C-102 breadcrumb
    bbox: tuple[float, ...] | None = None  # C-085 bounding box when the parser provides it
    char_start: int = 0  # offset in the canonical markdown (C-120)
    boundary: bool = False  # semantic chunking: start a new chunk here (S-02)


@dataclass
class ParsedDocument:
    source: str
    title: str
    elements: list[Element]
    parser: str
    language: str = ""
    page_count: int | None = None
    quality: float = 1.0  # C-088: 0-1 parse quality score
    quality_flags: list[str] = field(default_factory=list)
    attachments: list[SourceDocument] = field(default_factory=list)  # C-083

    def to_markdown(self) -> str:
        """Canonical text form (C-086): headings as #, tables kept as markdown."""
        lines = []
        for element in self.elements:
            if element.type in ("title", "heading"):
                lines.append(f"{'#' * max(1, min(element.level or 1, 6))} {element.text}")
            elif element.type == "list_item":
                lines.append(f"- {element.text}")
            elif element.type == "code":
                lines.append(f"```\n{element.text}\n```")
            else:
                lines.append(element.text)
        return "\n\n".join(lines)

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements)


class DocumentMetadata(BaseModel):
    """Validated metadata schema (C-100, C-108). Carried into every chunk (C-119)."""

    doc_id: str
    source: str
    collection: str
    title: str = ""
    url: str = ""
    author: str = ""
    created: str | None = None
    modified: str | None = None
    effective_date: str | None = None  # business effective date (C-103), ISO 8601
    doc_type: str = "other"  # C-105
    language: str = ""  # C-087
    acl: list[str] = Field(default_factory=lambda: ["*"])  # C-106
    tags: list[str] = Field(default_factory=list)
    department: str = ""  # C-103 business metadata
    region: str = ""
    sensitivity: Literal["public", "internal", "confidential", "restricted"] = "internal"  # C-095
    version: str = ""  # source version / etag
    content_hash: str = ""
    summary: str = ""  # C-104
    size_bytes: int = 0
    page_count: int | None = None
    pipeline_versions: dict[str, str] = Field(default_factory=dict)  # C-107
    extra: dict[str, str] = Field(default_factory=dict)

    def chunk_metadata(self) -> dict[str, str]:
        """Flat string metadata stored on each chunk (filterable fields are separate)."""
        base = {
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "department": self.department,
            "region": self.region,
            "sensitivity": self.sensitivity,
            "version": self.version,
        }
        return {k: v for k, v in {**base, **self.extra}.items() if v}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # what is embedded / searched (may carry a title prefix, C-121)
    index: int
    token_count: int
    kind: Literal["chunk", "parent"] = "chunk"
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    char_start: int = 0
    char_end: int = 0
    element_types: list[str] = field(default_factory=list)
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    context: str = ""  # contextual-retrieval enrichment (S-04)
    raw_text: str = ""  # text without prefixes, for display

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

"""Text cleaning (C-089, C-090, C-091) applied to every parsed element."""

import re
import unicodedata
from collections import Counter

from app.rag.ingestion.models import ParsedDocument

_MOJIBAKE = re.compile(r"[ÃÂâ][\x80-\xbf€™œ\"]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.!?:;\n])\n(?!\n|\s*[-*•\d|#])")


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252 ("â€™" -> "'")."""
    if not _MOJIBAKE.search(text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_text(text: str) -> str:
    """C-089: NFKC, mojibake repair, control characters and whitespace cleanup."""
    text = unicodedata.normalize("NFKC", fix_mojibake(text))
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def repair_lines(text: str) -> str:
    """C-091: join words hyphenated across line breaks and lines broken mid-sentence."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _SOFT_BREAK.sub(" ", text)


def remove_boilerplate(
    parsed: ParsedDocument, patterns: list[re.Pattern[str]], min_pages: int = 3
) -> int:
    """C-090: drop page furniture repeated across many pages (headers, footers, legal
    notices) and paragraphs matching configured disclaimer patterns. Returns the count."""
    pages = {e.page for e in parsed.elements if e.page}
    counts = Counter(
        e.text.strip().lower() for e in parsed.elements if e.page and len(e.text) < 200
    )
    repeated = (
        {text for text, n in counts.items() if n >= max(min_pages, len(pages) // 2)}
        if len(pages) >= min_pages
        else set()
    )
    kept, removed = [], 0
    for element in parsed.elements:
        lowered = element.text.strip().lower()
        if lowered in repeated or any(p.search(element.text) for p in patterns):
            removed += 1
            continue
        kept.append(element)
    parsed.elements = kept
    return removed


def clean_document(
    parsed: ParsedDocument, boilerplate_patterns: list[re.Pattern[str]] | None = None
) -> None:
    remove_boilerplate(parsed, boilerplate_patterns or [])
    for element in parsed.elements:
        text = element.text if element.type in ("code", "table") else repair_lines(element.text)
        element.text = normalize_text(text) if element.type != "code" else text.strip()
    parsed.elements = [e for e in parsed.elements if e.text]

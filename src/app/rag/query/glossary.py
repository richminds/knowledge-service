"""Domain glossary: acronym expansion (C-162) and keyword synonyms (S-08).

Placeholder file: config/glossary.yaml (GLOSSARY_PATH). Fill it with your organisation's
acronyms and synonyms; the code works with an empty or missing file.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Glossary:
    acronyms: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)

    def expand(self, query: str) -> str:
        """Append each known acronym's expansion: "HR policy" -> "HR (Human Resources) policy"."""

        def replace(match: re.Match[str]) -> str:
            word = match.group(0)
            expansion = self.acronyms.get(word) or self.acronyms.get(word.upper())
            if expansion and expansion.lower() not in query.lower():
                return f"{word} ({expansion})"
            return word

        return re.sub(r"\b[A-Za-z][A-Za-z0-9&]{1,9}\b", replace, query) if self.acronyms else query

    def keyword_variants(self, query: str) -> list[str]:
        """Synonyms of words in the query, for BM25 expansion."""
        words = {w.lower() for w in re.findall(r"[A-Za-z][\w-]+", query)}
        lowered = query.lower()
        variants: list[str] = []
        for term, alternatives in self.synonyms.items():
            if term.lower() in words or (" " in term and term.lower() in lowered):
                variants.extend(a for a in alternatives if a.lower() not in lowered)
        return variants


@lru_cache(maxsize=4)
def load_glossary(path: str) -> Glossary:
    file = Path(path)
    if not file.is_file():
        return Glossary()
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("glossary_invalid", path=path, error=str(exc))
        return Glossary()
    return Glossary(
        acronyms={str(k): str(v) for k, v in (data.get("acronyms") or {}).items()},
        synonyms={
            str(k): [str(x) for x in (v or [])] for k, v in (data.get("synonyms") or {}).items()
        },
    )

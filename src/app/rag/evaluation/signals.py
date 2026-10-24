"""Answer quality signals computed from the text alone (no LLM call, every response)."""

import re

from app.models.schemas import QualitySignals
from app.rag.generation.prompts import NO_CONTEXT_ANSWER

_CITATION = re.compile(r"\[(\d+)\]")
# Statement boundaries: sentence punctuation or line breaks (bullet lists).
_STATEMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_REFUSAL = re.compile(
    r"(\bI (?:do not|don't|cannot|can't|could not|couldn't) (?:know|find|answer)"
    r"|\bnot (?:contained|mentioned|provided|available|included)"
    r" in the (?:context|sources|passages)"
    r"|\bno (?:relevant )?information (?:about|on|regarding))",
    re.IGNORECASE,
)
_MIN_STATEMENT_WORDS = 4  # headings / fragments don't need citations
# "...library. [1]" - a marker after the full stop belongs to the sentence before it.
_TRAILING_CITATIONS = re.compile(r"([.!?])[ \t]*((?:\[\d+\])+)")


def statements(answer: str) -> list[str]:
    """Claims that should carry a citation. Lead-ins ending in ':' introduce a list rather
    than state a fact, so they are not counted."""
    text = _TRAILING_CITATIONS.sub(r" \2\1", answer.strip())
    parts = (p.strip(" -*\t") for p in _STATEMENT_SPLIT.split(text))
    return [
        p
        for p in parts
        if len(_CITATION.sub("", p).split()) >= _MIN_STATEMENT_WORDS and not p.endswith(":")
    ]


def compute_signals(answer: str, contexts_retrieved: int) -> QualitySignals:
    cited = sorted({int(n) for n in _CITATION.findall(answer)})
    valid = [n for n in cited if 1 <= n <= contexts_retrieved]
    claims = statements(answer)
    answered = bool(answer.strip()) and answer != NO_CONTEXT_ANSWER
    if answered and _REFUSAL.search(answer) and not valid:
        answered = False
    return QualitySignals(
        answered=answered,
        contexts_retrieved=contexts_retrieved,
        cited_passages=valid,
        invalid_citations=[n for n in cited if n not in valid],
        citation_coverage=(
            round(sum(1 for c in claims if _CITATION.search(c)) / len(claims), 3) if claims else 0.0
        ),
        cited_context_ratio=(
            round(len(valid) / contexts_retrieved, 3) if contexts_retrieved else 0.0
        ),
    )

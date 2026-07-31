from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.documents import Document


class CitationValidation(TypedDict):
    valid: bool
    cited_numbers: list[int]
    missing_citations: bool
    invalid_citations: list[int]
    available_citations: list[int]


def validate_citations(answer: str, chunks: list[Document]) -> CitationValidation:
    """Validate that answer citations refer to available context chunks.

    Implementation:
        The function extracts citation markers like `[1]` and `[2]`, compares
        them with the number of selected chunks, and reports missing or invalid
        citations.

    Usage:
        The query graph runs this after answer generation.

    How it helps other functions:
        Citation validation gives the caller a quality signal that can be logged,
        displayed, or used by evaluation to detect ungrounded answers.
    """
    cited_numbers = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})
    available = list(range(1, len(chunks) + 1))
    invalid = [number for number in cited_numbers if number not in available]
    return {
        "valid": bool(cited_numbers) and not invalid,
        "cited_numbers": cited_numbers,
        "missing_citations": not bool(cited_numbers),
        "invalid_citations": invalid,
        "available_citations": available,
    }

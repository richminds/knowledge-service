"""Lightweight PII detection/masking (regex).

Used for log/trace masking (C-196), the output guardrail (C-195) and ingestion-time
masking (C-094, PII_MODE=regex) — see app.rag.ingestion.governance.
"""

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<![\w-])\+?\d{1,3}?[ .-]?\(?\d{2,4}\)?[ .-]\d{3,4}[ .-]\d{3,4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    checksum = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        checksum += n
    return len(nums) >= 13 and checksum % 10 == 0


def find_pii(text: str) -> list[tuple[str, str]]:
    """(category, matched text) pairs."""
    found = []
    for category, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if category == "CREDIT_CARD" and not _luhn_ok(value):
                continue
            found.append((category, value))
    return found


def mask_pii(text: str) -> str:
    for category, value in sorted(find_pii(text), key=lambda f: -len(f[1])):
        text = text.replace(value, f"[{category}]")
    return text


def mask_pii_processor(logger, method_name, event_dict: dict) -> dict:
    """structlog processor: mask PII in every string value of a log event."""
    for key, value in event_dict.items():
        if isinstance(value, str) and key not in ("timestamp", "level", "logger"):
            event_dict[key] = mask_pii(value)
    return event_dict

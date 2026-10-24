"""Guardrails: input (C-156), retrieved content (C-194, C-097) and output (C-195).

CONTENT_SAFETY_MODE:
  off        - no checks
  heuristic  - local pattern checks (prompt injection / jailbreak phrases, leaked PII; default)
"""

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.pii import find_pii

logger = get_logger(__name__)

# Phrases typical of jailbreaks / injected instructions. Deliberately conservative: a match
# on retrieved content only drops that passage; on user input it blocks the request.
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules)",
    r"disregard (?:all |any |the )?(?:previous|prior|above) (?:instructions|rules)",
    r"(?:reveal|print|show|repeat) (?:your|the) (?:system|hidden|initial) (?:prompt|instructions)",
    r"you are now (?:in )?(?:developer|dan|jailbreak|unrestricted) mode",
    r"\bdo anything now\b",
    r"pretend (?:that )?you (?:have no|are not bound by) (?:rules|restrictions|guidelines)",
    r"<\s*/?\s*(?:system|assistant)\s*>",
    r"\bnew instructions\s*:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class SafetyVerdict:
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)


def heuristic_injection(text: str) -> str | None:
    match = _INJECTION_RE.search(text)
    return match.group(0) if match else None


class ContentSafety:
    def __init__(self, mode: str = "heuristic", block_output_pii: bool = True) -> None:
        self._mode = mode
        self._block_output_pii = block_output_pii

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    async def check_input(self, text: str) -> SafetyVerdict:
        """C-156: block jailbreak / prompt-injection attempts."""
        if not self.enabled:
            return SafetyVerdict()
        reasons = []
        if phrase := heuristic_injection(text):
            reasons.append(f"injection_pattern:{phrase[:40]}")
        return SafetyVerdict(blocked=bool(reasons), reasons=reasons)

    async def flag_documents(self, texts: list[str]) -> list[bool]:
        """C-194: True for passages that contain instructions aimed at the model."""
        if not self.enabled or not texts:
            return [False] * len(texts)
        return [heuristic_injection(t) is not None for t in texts]

    async def check_output(self, text: str, allowed_pii: str = "") -> SafetyVerdict:
        """C-195: PII that did not come from the question/sources."""
        if not self.enabled:
            return SafetyVerdict()
        reasons = []
        if self._block_output_pii:
            leaked = [v for _, v in find_pii(text) if v not in allowed_pii]
            if leaked:
                reasons.append(f"pii:{len(leaked)}")
        return SafetyVerdict(blocked=bool(reasons), reasons=reasons)

    async def close(self) -> None:
        return None

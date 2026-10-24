"""Guardrails: input (C-156), retrieved content (C-194, C-097) and output (C-195).

CONTENT_SAFETY_MODE:
  off        - no checks
  heuristic  - local pattern checks only (no external service; default)
  azure      - Azure AI Content Safety: Prompt Shields (jailbreak + indirect injection in
               documents) and harm-category analysis, plus the heuristics
Placeholders: CONTENT_SAFETY_ENDPOINT (https://<name>.cognitiveservices.azure.com) and
optionally CONTENT_SAFETY_API_KEY (blank = Entra ID / managed identity).
"""

import re
from dataclasses import dataclass, field

import httpx

from app.core.azure_auth import COGNITIVE_SERVICES_SCOPE, get_azure_credential
from app.core.logging import get_logger
from app.core.pii import find_pii

logger = get_logger(__name__)

_API_VERSION = "2024-09-01"
_MAX_CHARS = 10_000
_HARM_CATEGORIES = ["Hate", "SelfHarm", "Sexual", "Violence"]

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
    def __init__(
        self,
        mode: str = "heuristic",
        endpoint: str = "",
        api_key: str = "",
        severity_threshold: int = 4,
        block_output_pii: bool = True,
    ) -> None:
        self._mode = mode
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._threshold = severity_threshold
        self._block_output_pii = block_output_pii
        self._http = httpx.AsyncClient(timeout=10.0) if mode == "azure" and endpoint else None

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    async def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Ocp-Apim-Subscription-Key": self._api_key}
        token = await get_azure_credential().get_token(COGNITIVE_SERVICES_SCOPE)
        return {"Authorization": f"Bearer {token.token}"}

    async def _call(self, operation: str, body: dict) -> dict | None:
        if self._http is None:
            return None
        try:
            response = await self._http.post(
                f"{self._endpoint}/contentsafety/text:{operation}",
                params={"api-version": _API_VERSION},
                json=body,
                headers=await self._headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - fail open on the service, keep heuristics
            logger.warning("content_safety_unavailable", operation=operation, error=str(exc))
            return None

    async def _harm(self, text: str) -> list[str]:
        result = await self._call(
            "analyze",
            {
                "text": text[:_MAX_CHARS],
                "categories": _HARM_CATEGORIES,
                "outputType": "FourSeverityLevels",
            },
        )
        if not result:
            return []
        return [
            f"harm:{c['category']}"
            for c in result.get("categoriesAnalysis", [])
            if c.get("severity", 0) >= self._threshold
        ]

    async def check_input(self, text: str) -> SafetyVerdict:
        """C-156: block jailbreak / prompt-injection attempts and harmful requests."""
        if not self.enabled:
            return SafetyVerdict()
        reasons = []
        if phrase := heuristic_injection(text):
            reasons.append(f"injection_pattern:{phrase[:40]}")
        if self._http is not None:
            shield = await self._call(
                "shieldPrompt", {"userPrompt": text[:_MAX_CHARS], "documents": []}
            )
            if shield and shield.get("userPromptAnalysis", {}).get("attackDetected"):
                reasons.append("prompt_shield:jailbreak")
            reasons += await self._harm(text)
        return SafetyVerdict(blocked=bool(reasons), reasons=reasons)

    async def flag_documents(self, texts: list[str]) -> list[bool]:
        """C-194: True for passages that contain instructions aimed at the model."""
        if not self.enabled or not texts:
            return [False] * len(texts)
        flags = [heuristic_injection(t) is not None for t in texts]
        if self._http is not None:
            docs = [t[: _MAX_CHARS // max(1, len(texts))] for t in texts]
            shield = await self._call("shieldPrompt", {"userPrompt": "", "documents": docs})
            if shield:
                for i, analysis in enumerate(shield.get("documentsAnalysis", [])):
                    if i < len(flags) and analysis.get("attackDetected"):
                        flags[i] = True
        return flags

    async def check_output(self, text: str, allowed_pii: str = "") -> SafetyVerdict:
        """C-195: harmful content, and PII that did not come from the question/sources."""
        if not self.enabled:
            return SafetyVerdict()
        reasons = []
        if self._block_output_pii:
            leaked = [v for _, v in find_pii(text) if v not in allowed_pii]
            if leaked:
                reasons.append(f"pii:{len(leaked)}")
        if self._http is not None:
            reasons += await self._harm(text)
        return SafetyVerdict(blocked=bool(reasons), reasons=reasons)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

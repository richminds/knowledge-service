"""Ingestion governance (C-070, C-071, C-092..C-098).

Decides per document: ingest, skip (unchanged/duplicate) or quarantine (too large,
malware, unparseable, low quality, suspected prompt injection, restricted, expired) and
applies PII masking. Quarantined files are kept in the `quarantine/` zone for review.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from app.core.azure_auth import COGNITIVE_SERVICES_SCOPE, get_azure_credential
from app.core.logging import get_logger
from app.core.pii import mask_pii
from app.rag.ingestion.models import ParsedDocument, SourceDocument
from app.rag.safety import heuristic_injection

logger = get_logger(__name__)

SENSITIVITY_ORDER = ["public", "internal", "confidential", "restricted"]
_MARKERS = {
    "restricted": re.compile(
        r"\b(strictly confidential|restricted|top secret|legal privilege)\b", re.I
    ),
    "confidential": re.compile(r"\b(confidential|not for (external )?distribution)\b", re.I),
    "internal": re.compile(r"\b(internal use only|for internal use|official use only)\b", re.I),
    "public": re.compile(r"\b(public disclosure authori[sz]ed|publicly available)\b", re.I),
}
# Purview / source labels -> our levels (extend for your label taxonomy).
LABEL_MAP = {
    "public": "public",
    "general": "internal",
    "internal": "internal",
    "official use only": "internal",
    "confidential": "confidential",
    "highly confidential": "restricted",
    "strictly confidential": "restricted",
    "restricted": "restricted",
}


@dataclass
class GovernancePolicy:
    max_bytes: int = 100 * 1024 * 1024  # C-070
    min_text_chars: int = 50  # C-096
    min_parse_quality: float = 0.4  # C-096 / C-088
    max_sensitivity: str = "confidential"  # C-095: anything above is quarantined
    injection_action: str = "quarantine"  # C-097: quarantine | flag
    pii_mode: str = "off"  # C-094: off | regex | azure_language
    pii_action: str = "mask"  # mask | flag
    pii_endpoint: str = ""  # Azure AI Language endpoint
    pii_api_key: str = ""
    retention_max_age_days: int = 0  # C-098: 0 = no age limit
    near_duplicate_threshold: float = 0.9  # C-092
    boilerplate_patterns: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str = "ingest"  # ingest | skip | quarantine
    reasons: list[str] = field(default_factory=list)
    sensitivity: str = "internal"
    pii_found: int = 0
    duplicate_of: str = ""

    def quarantine(self, reason: str) -> "Decision":
        self.action = "quarantine"
        self.reasons.append(reason)
        return self


# -- pre-parse checks -------------------------------------------------------------------------
def check_raw(document: SourceDocument, policy: GovernancePolicy) -> Decision:
    decision = Decision()
    if len(document.content) > policy.max_bytes:
        return decision.quarantine(f"too_large:{len(document.content)}")
    scan = document.metadata.get("malware_scan", "")
    if scan and scan.lower() not in ("no threats found", "clean"):
        return decision.quarantine(f"malware_scan:{scan}")  # C-071 (Defender for Storage)
    if document.metadata.get("legal_hold", "").lower() == "true":
        return decision.quarantine("legal_hold")  # C-098
    expires = document.metadata.get("expires")
    if expires and expires[:10] < datetime.now(UTC).date().isoformat():
        return decision.quarantine("expired")  # C-098
    if policy.retention_max_age_days and document.last_modified:
        modified = datetime.fromisoformat(document.last_modified.replace("Z", "+00:00"))
        if modified.tzinfo is None:
            modified = modified.replace(tzinfo=UTC)
        if datetime.now(UTC) - modified > timedelta(days=policy.retention_max_age_days):
            return decision.quarantine("retention_age")
    return decision


# -- post-parse checks --------------------------------------------------------------------------
def classify_sensitivity(document: SourceDocument, text: str) -> str:
    """C-095: source label (e.g. Purview) first, then markers in the opening text."""
    label = LABEL_MAP.get(document.sensitivity_label.strip().lower())
    if label:
        return label
    head = text[:3000]
    for level in ("restricted", "confidential", "internal", "public"):
        if _MARKERS[level].search(head):
            return level
    return "internal"


def check_parsed(
    document: SourceDocument, parsed: ParsedDocument, policy: GovernancePolicy
) -> Decision:
    decision = Decision()
    text = parsed.text
    if len(text.strip()) < policy.min_text_chars:
        return decision.quarantine("no_or_little_text")
    if parsed.quality < policy.min_parse_quality:
        return decision.quarantine(
            f"low_parse_quality:{parsed.quality}:{','.join(parsed.quality_flags)}"
        )
    decision.sensitivity = classify_sensitivity(document, text)
    if SENSITIVITY_ORDER.index(decision.sensitivity) > SENSITIVITY_ORDER.index(
        policy.max_sensitivity
    ):
        return decision.quarantine(f"sensitivity:{decision.sensitivity}")
    if phrase := heuristic_injection(text):
        decision.reasons.append(f"injection_suspect:{phrase[:40]}")
        if policy.injection_action == "quarantine":
            return decision.quarantine("prompt_injection")
    return decision


# -- PII (C-094) -----------------------------------------------------------------------------------
class PiiMasker:
    def __init__(self, policy: GovernancePolicy) -> None:
        self._policy = policy
        self._http = (
            httpx.AsyncClient(timeout=30.0) if policy.pii_mode == "azure_language" else None
        )

    @staticmethod
    async def _entra_token() -> str:
        return (await get_azure_credential().get_token(COGNITIVE_SERVICES_SCOPE)).token

    async def _azure_redact(self, text: str) -> tuple[str, int]:
        headers = (
            {"Ocp-Apim-Subscription-Key": self._policy.pii_api_key}
            if self._policy.pii_api_key
            else {"Authorization": f"Bearer {await self._entra_token()}"}
        )
        redacted_parts, found = [], 0
        for start in range(0, len(text), 5000):  # service limit per document
            response = await self._http.post(
                f"{self._policy.pii_endpoint.rstrip('/')}/language/:analyze-text",
                params={"api-version": "2023-04-01"},
                headers=headers,
                json={
                    "kind": "PiiEntityRecognition",
                    "analysisInput": {
                        "documents": [{"id": "1", "text": text[start : start + 5000]}]
                    },
                },
            )
            response.raise_for_status()
            doc = response.json()["results"]["documents"][0]
            redacted_parts.append(doc["redactedText"])
            found += len(doc.get("entities", []))
        return "".join(redacted_parts), found

    async def apply(self, parsed: ParsedDocument) -> int:
        mode = self._policy.pii_mode
        if mode == "off":
            return 0
        found = 0
        for element in parsed.elements:
            if mode == "azure_language" and self._http is not None:
                masked, count = await self._azure_redact(element.text)
            else:
                masked = mask_pii(element.text)
                count = int(masked != element.text)
            found += count
            if self._policy.pii_action == "mask":
                element.text = masked
        return found


# -- near-duplicates (C-092) ----------------------------------------------------------------------
_SEEDS = [hashlib.sha256(str(i).encode()).digest()[:8] for i in range(64)]


def minhash(text: str, shingle: int = 5) -> list[int]:
    words = re.findall(r"\w+", text.lower())
    shingles = {" ".join(words[i : i + shingle]) for i in range(max(1, len(words) - shingle + 1))}
    if not shingles:
        return []
    return [
        min(
            int.from_bytes(hashlib.blake2b(s.encode(), key=seed, digest_size=8).digest(), "big")
            for s in shingles
        )
        for seed in _SEEDS
    ]


def similarity(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b, strict=False)) / len(a)

"""Every LLM and embedding call this service makes: through the API gateway, to the LLM gateway.

This service holds no provider credential and imports no provider SDK. ChatClient and
EmbeddingClient post to the LLM gateway's ``POST /v1/llm`` and ``POST /v1/embeddings``
with the vendored client in llm_gateway_sdk.py. Model choice, fallbacks, retries,
provider keys and cost tracking are the LLM gateway's business. Same design as
makemerich-backend's app/agents/llm_gateway.py.

**Where it goes.** ``LLM_GATEWAY_URL`` when set (e.g. straight at a local llm-gateway),
otherwise ``API_GATEWAY_URL`` + ``/api/llm``: the API gateway strips the prefix and
forwards, and is the single place where authentication, per-user rate limits, budgets and
metering are enforced.

**Which credential goes out.** The caller's own bearer token, read per request from
app/core/caller_context.py, so one pooled client serves a different user on every call.
With no caller bound (CLI ingestion, scripts, AUTH_MODE=none) the service key
``LLM_GATEWAY_API_KEY`` is sent as ``X-API-Key`` when configured; otherwise the call goes
out with no credential and the gateway refuses it, a visible failure rather than a quiet
mis-attribution.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from typing import Any

import httpx

from app.core.caller_context import get_caller_token
from app.rag.generation.llm_gateway_sdk import GatewayError, RemoteLLMClient

# The gateway's own category vocabulary (llm-gateway features/classifier.py). Set
# explicitly rather than left to the classifier, so routing (and cost) is predictable.
CATEGORY_FACTUAL_QA = "factual_qa"  # grounded answers, extraction, grading
CATEGORY_CONVERSATIONAL = "conversational"  # chit-chat replies
CATEGORY_SUMMARIZATION = "summarization"

_client: RemoteLLMClient | None = None


class LLMUnavailableError(RuntimeError):
    """No gateway is configured: the one way an LLM call fails before leaving the process."""


class _CallerCredential(httpx.Auth):
    """Attach the current caller's bearer token (or the service key) to each request."""

    def __init__(self, service_api_key: str = "") -> None:
        self._service_api_key = service_api_key

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        token = get_caller_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        elif self._service_api_key:
            request.headers["X-API-Key"] = self._service_api_key
        yield request


def gateway_base_url(llm_gateway_url: str, api_gateway_url: str) -> str:
    if llm_gateway_url:
        return llm_gateway_url.rstrip("/")
    if api_gateway_url:
        return f"{api_gateway_url.rstrip('/')}/api/llm"
    return ""


def get_llm_client(
    llm_gateway_url: str, api_gateway_url: str, api_key: str = "", timeout: float = 120.0
) -> RemoteLLMClient:
    """The process-wide client, built on first use: one connection pool."""
    global _client
    if _client is None:
        base_url = gateway_base_url(llm_gateway_url, api_gateway_url)
        if not base_url:
            raise LLMUnavailableError(
                "Neither LLM_GATEWAY_URL nor API_GATEWAY_URL is set; every LLM call goes "
                "through the gateway and there is nowhere to send this one."
            )
        _client = RemoteLLMClient(
            base_url=base_url, timeout=timeout, auth=_CallerCredential(api_key)
        )
    return _client


def reset_llm_client() -> None:
    """Drop the singleton so the next call rebuilds it (tests, settings reloads)."""
    global _client
    _client = None


async def aclose_llm_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply, tolerant of the ways models wrap it.

    Whichever model the gateway picks may fence the JSON or prefix a sentence: strip a
    fence if there is one, then take the outermost ``{...}``. Raises ``ValueError`` when
    there is none or it does not parse.
    """
    body = text.strip()
    fenced = _FENCE_RE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in the model's reply")
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"model reply is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model reply is JSON but not an object")
    return parsed


__all__ = [
    "CATEGORY_CONVERSATIONAL",
    "CATEGORY_FACTUAL_QA",
    "CATEGORY_SUMMARIZATION",
    "GatewayError",
    "LLMUnavailableError",
    "aclose_llm_client",
    "extract_json_object",
    "gateway_base_url",
    "get_llm_client",
    "reset_llm_client",
]

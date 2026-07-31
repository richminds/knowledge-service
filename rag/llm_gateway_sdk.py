"""Vendored HTTP client for the LLM Gateway service.

This is an unmodified copy of ``sdk/client.py`` from the ``llm-gateway``
project, dropped in here per that module's own docstring: "Drop this into any
application that should reach an LLM through the gateway rather than talking
to a provider directly." This is precisely that drop-in — it is what lets the
Knowledge Service hold zero provider credentials and import zero provider
SDKs while still doing every embedding and chat call through a real LLM
Gateway deployment, over HTTP, on the request path (see
``rag/llm_gateway_client.py`` for the singletons built from it).

Kept as a single vendored file rather than a package dependency because
llm-gateway is not published as an installable package — this mirrors how the
gateway's own docs describe integrating it into a calling application.

Usage::

    from rag.llm_gateway_sdk import RemoteLLMClient

    client = RemoteLLMClient(
        base_url="https://llm-gateway.internal",
        api_key="sk-live-...",       # identifies your application
    )
    response = await client.chat(
        [{"role": "user", "content": "Summarise this incident report."}],
        caller="support-triage",
    )
    print(response.content, response.total_tokens)

    async for event in client.stream(messages, caller="support-triage"):
        print(event.content, end="", flush=True)

    await client.aclose()

``GatewayEmbeddingsClient`` implements LangChain's ``Embeddings`` interface, so
it drops into an existing RAG pipeline unchanged.

The application never holds a provider API key: the gateway owns those. It
holds one gateway credential, which is what makes rotation and per-application
quotas possible in the first place.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0


@dataclass
class LLMResponse:
    """Mirror of ``features.client.LLMResponse`` — same fields, same property."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw: dict[str, Any] = field(default_factory=dict)
    # Classified request category (features/classifier.py) that selected the
    # chain this response was served from. "default" when unclassified.
    category: str = "default"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StreamEvent:
    """One event from ``RemoteLLMClient.stream``.

    Token events carry ``content``. The terminal event has ``is_final=True``
    and either usage totals or ``error``.
    """

    content: str = ""
    model: str = ""
    finish_reason: str = ""
    is_final: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    # Populated only on the final event, mirroring the token-count fields.
    category: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GatewayError(Exception):
    """Raised when the gateway returns a non-2xx response.

    ``status_code`` distinguishes the cases a caller may want to handle
    differently: 503 means every model failed (retryable), 429 means the caller
    is throttled, 500 means the gateway itself is misconfigured.
    """

    def __init__(self, status_code: int, code: str, message: str, request_id: str = "") -> None:
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        suffix = f" (id={request_id})" if request_id else ""
        super().__init__(f"[{status_code} {code}] {message}" + suffix)


def _raise_for_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    code, message, request_id = "http_error", response.text[:500], ""
    try:
        payload = response.json()
        err = payload.get("error") or {}
        code = err.get("code", code)
        message = err.get("message", payload.get("detail", message))
        request_id = err.get("request_id", "")
    except Exception:  # noqa: BLE001 — non-JSON body; keep the raw text
        pass
    raise GatewayError(response.status_code, code, str(message), request_id)


class RemoteLLMClient:
    """Drop-in replacement for the in-process ``LLMClient``, over HTTP.

    One instance owns one connection pool — build it once per process and reuse
    it, exactly as with the in-process client.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        jwt: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        if jwt:
            headers["Authorization"] = f"Bearer {jwt}"

        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=timeout
        )
        self._owns_client = client is None

    # ------------------------------------------------------------------ chat

    async def chat(
        self,
        messages: list[dict[str, str]],
        caller: str = "unknown",
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Same signature as the in-process client. Raises ``GatewayError``.

        ``category`` overrides the gateway's automatic classification (see
        ``features/classifier.py``) — leave unset to let the gateway classify
        the prompt itself. Extra kwargs are forwarded to the provider through
        the gateway's ``extra`` field (e.g. ``top_p``, ``response_format``).
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "caller": caller,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "category": category,
            "extra": kwargs,
        }
        response = await self._client.post("/v1/llm", json=payload)
        _raise_for_error(response)
        data = response.json()
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["content"],
            model=data["model"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=data.get("finish_reason", ""),
            raw=data,
            category=data.get("category", "default"),
        )

    # ------------------------------------------------------------ streaming

    async def stream(
        self,
        messages: list[dict[str, str]],
        caller: str = "unknown",
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield tokens as they arrive from the gateway.

        The final event has ``is_final=True`` and carries usage totals, or
        ``error`` if the stream failed. Errors are delivered as events rather
        than exceptions: the HTTP response has already begun by the time
        anything can go wrong, so there is no status code left to raise on.

        Usage::

            async for event in client.stream(messages, caller="chat-ui"):
                if event.error:
                    break
                print(event.content, end="", flush=True)
        """
        payload = {
            "messages": messages,
            "caller": caller,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "category": category,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/llm", json=payload) as response:
            if response.status_code >= 400:
                # Rejections that happen before streaming starts (401, 413, 422)
                # still arrive as a normal error body.
                await response.aread()
                _raise_for_error(response)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = frame.get("usage") or {}
                yield StreamEvent(
                    content=frame.get("content", ""),
                    model=frame.get("model", ""),
                    finish_reason=frame.get("finish_reason", ""),
                    is_final="usage" in frame or "error" in frame,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    error=frame.get("error", ""),
                    category=frame.get("category", ""),
                )

    # ------------------------------------------------------------ embeddings

    async def embed(
        self, texts: list[str], input_type: str = "document", caller: str = "unknown"
    ) -> list[list[float]]:
        response = await self._client.post(
            "/v1/embeddings",
            json={"input": texts, "input_type": input_type, "caller": caller},
        )
        _raise_for_error(response)
        return response.json()["embeddings"]

    # ------------------------------------------------------------- telemetry

    async def stats(self, caller: str | None = None) -> dict[str, Any]:
        params = {"caller": caller} if caller else None
        response = await self._client.get("/v1/stats", params=params)
        _raise_for_error(response)
        return response.json()

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/health")
        return response.json()

    # ------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> RemoteLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


class GatewayEmbeddingsClient:
    """LangChain-compatible ``Embeddings`` served by the remote gateway.

    Implements the synchronous LangChain interface (``embed_documents`` /
    ``embed_query``) because that is what RAG pipelines expect; async variants
    are provided for callers already inside an event loop.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        with httpx.Client(
            base_url=self._base_url, headers=self._headers, timeout=self._timeout
        ) as c:
            response = c.post(
                "/v1/embeddings", json={"input": texts, "input_type": input_type}
            )
        _raise_for_error(response)
        return response.json()["embeddings"]

    async def _apost(self, texts: list[str], input_type: str) -> list[list[float]]:
        async with httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers, timeout=self._timeout
        ) as c:
            response = await c.post(
                "/v1/embeddings", json={"input": texts, "input_type": input_type}
            )
        _raise_for_error(response)
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._post(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "query")[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._apost(texts, "document")

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._apost([text], "query"))[0]

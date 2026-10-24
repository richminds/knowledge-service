import json
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.core.azure_auth import build_openai_client
from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.core.usage import record_usage
from app.rag.generation.prompts import PromptTemplate, format_context, load_prompt
from app.rag.vectorstore.base import VectorSearchResult

logger = get_logger(__name__)

# Errors after which the secondary deployment is tried (C-190). The SDK has already
# retried them (max_retries) against the primary.
_FAILOVER_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# Reasoning deployments (o-series, gpt-5.x) take `max_completion_tokens` - which also pays
# for hidden reasoning tokens - and only the default temperature. Recognised by name; any
# other deployment that rejects `max_tokens` is learned on its first call.
_REASONING_NAME = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)
REASONING_TOKEN_HEADROOM = 4096


def _for_reasoning(kwargs: dict) -> dict:
    adapted = {k: v for k, v in kwargs.items() if k != "temperature"}
    if "max_tokens" in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens") + REASONING_TOKEN_HEADROOM
    return adapted


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_NAME.match(model))


def reasoning_params(kwargs: dict) -> dict:
    """`kwargs` as a reasoning deployment accepts them (no temperature, max_completion_tokens)."""
    return _for_reasoning(kwargs)


# (role, content) pairs of prior turns, oldest first. Only "user"/"assistant" roles.
History = Sequence[tuple[str, str]]

# Answer deployment for the current request (admin playground model comparison). A context
# variable, so concurrent requests don't mix and the strategies need no extra parameter.
_answer_model: ContextVar[str | None] = ContextVar("answer_model", default=None)


@contextmanager
def answer_model(deployment: str | None) -> Iterator[None]:
    """Generate answers with `deployment` instead of the configured chat deployment."""
    token = _answer_model.set(deployment or None)
    try:
        yield
    finally:
        _answer_model.reset(token)


def _record(purpose: str, model: str, usage) -> None:
    if usage is not None:
        record_usage(purpose, model, usage.prompt_tokens or 0, usage.completion_tokens or 0)


class ChatClient:
    """Async Azure OpenAI chat client: grounded answers (sync + streaming), JSON helper
    calls on the cheaper utility deployment, and failover to a secondary endpoint."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        utility_deployment: str | None = None,
        fallback_endpoint: str = "",
        fallback_api_key: str = "",
        fallback_deployment: str = "",
        prompt_versions: dict[str, str] | None = None,
        organisation: str = "",
    ) -> None:
        if not endpoint:
            raise GenerationError("Azure OpenAI endpoint is not configured")
        self._client = build_openai_client(endpoint, api_key, api_version)
        self._deployment = deployment
        self._utility_deployment = utility_deployment or deployment
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._fallback = (
            build_openai_client(fallback_endpoint, fallback_api_key, api_version)
            if fallback_endpoint
            else None
        )
        self._fallback_deployment = fallback_deployment or deployment
        self._prompt_versions = prompt_versions or {}
        self._organisation = organisation
        self._reasoning: set[str] = set()  # deployments learned to need reasoning parameters

    @property
    def answer_deployment(self) -> str:
        return _answer_model.get() or self._deployment

    def prompt(self, name: str) -> PromptTemplate:
        return load_prompt(name, self._prompt_versions.get(name))

    def _params(self, model: str, kwargs: dict) -> dict:
        if model in self._reasoning or _REASONING_NAME.match(model):
            return _for_reasoning(kwargs)
        return kwargs

    async def _call(self, client, model: str, kwargs: dict):
        try:
            return await client.chat.completions.create(model=model, **self._params(model, kwargs))
        except BadRequestError as exc:
            if getattr(exc, "param", None) not in ("max_tokens", "temperature") or (
                model in self._reasoning
            ):
                raise
            logger.info("llm_reasoning_parameters", model=model, param=exc.param)
            self._reasoning.add(model)
            return await client.chat.completions.create(model=model, **_for_reasoning(kwargs))

    async def _create(self, model: str, **kwargs):
        try:
            return await self._call(self._client, model, kwargs)
        except _FAILOVER_ERRORS as exc:
            if self._fallback is None:
                raise
            fallback_model = self._fallback_deployment if model == self._deployment else model
            logger.warning("llm_failover", model=model, error=type(exc).__name__)
            return await self._call(self._fallback, fallback_model, kwargs)

    def _answer_messages(
        self, query: str, contexts: list[VectorSearchResult], history: History, prompt: str
    ) -> list[dict[str, str]]:
        system, user = self.prompt(prompt).render(
            context=format_context(contexts), question=query, organisation=self._organisation
        )
        return [
            {"role": "system", "content": system},
            *({"role": role, "content": content} for role, content in history),
            {"role": "user", "content": user},
        ]

    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        deployment: str | None = None,
        max_tokens: int = 1500,
    ) -> dict:
        """Structured helper call (graph extraction, agent routing/grading, judging) -> JSON.

        `purpose` labels the call in token metrics and lets tests fake each call type.
        """
        model = deployment or self._utility_deployment
        try:
            response = await self._create(
                model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            _record(purpose, model, response.usage)
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM call '{purpose}' failed: {exc}") from exc

    async def complete_text(self, prompt: str, *, purpose: str, **values: str) -> str:
        """Plain completion from a registry prompt on the utility model (e.g. chit-chat)."""
        system, user = self.prompt(prompt).render(organisation=self._organisation, **values)
        try:
            response = await self._create(
                self._utility_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=300,
            )
            _record(purpose, self._utility_deployment, response.usage)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM call '{purpose}' failed: {exc}") from exc

    async def generate_answer(
        self,
        query: str,
        contexts: list[VectorSearchResult],
        history: History = (),
        prompt: str = "rag_answer",
    ) -> str:
        model = self.answer_deployment
        try:
            response = await self._create(
                model,
                messages=self._answer_messages(query, contexts, history, prompt),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Chat completion failed: {exc}") from exc
        _record("answer", model, response.usage)
        return response.choices[0].message.content or ""

    async def stream_answer(
        self,
        query: str,
        contexts: list[VectorSearchResult],
        history: History = (),
        prompt: str = "rag_answer",
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they are generated (used by the SSE chat endpoint)."""
        model = self.answer_deployment
        try:
            stream = await self._create(
                model,
                messages=self._answer_messages(query, contexts, history, prompt),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
                # The final chunk then carries token usage (no choices).
                stream_options={"include_usage": True},
            )
            async for event in stream:
                if event.usage is not None:
                    _record("answer", model, event.usage)
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Chat completion stream failed: {exc}") from exc

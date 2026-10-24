from collections.abc import AsyncIterator, Sequence

from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.core.usage import record_usage
from app.rag.generation.llm_gateway import (
    CATEGORY_CONVERSATIONAL,
    CATEGORY_FACTUAL_QA,
    extract_json_object,
)
from app.rag.generation.llm_gateway_sdk import RemoteLLMClient
from app.rag.generation.prompts import PromptTemplate, format_context, load_prompt
from app.rag.vectorstore.base import VectorSearchResult

logger = get_logger(__name__)

# (role, content) pairs of prior turns, oldest first. Only "user"/"assistant" roles.
History = Sequence[tuple[str, str]]


def _record(purpose: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    if prompt_tokens or completion_tokens:
        record_usage(purpose, model or "llm-gateway", prompt_tokens, completion_tokens)


class ChatClient:
    """Chat client over the LLM gateway: grounded answers (sync + streaming) and JSON helper
    calls. The gateway picks the model (and fails over); this client only says which
    category of prompt each call is and labels it with `caller` for per-feature stats."""

    def __init__(
        self,
        client: RemoteLLMClient,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        caller: str = "knowledge-service",
        prompt_versions: dict[str, str] | None = None,
        organisation: str = "",
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._caller = caller
        self._prompt_versions = prompt_versions or {}
        self._organisation = organisation

    def prompt(self, name: str) -> PromptTemplate:
        return load_prompt(name, self._prompt_versions.get(name))

    def _caller_for(self, purpose: str) -> str:
        return f"{self._caller}:{purpose}"

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
        max_tokens: int = 1500,
    ) -> dict:
        """Structured helper call (graph extraction, agent routing/grading, judging) -> JSON.

        `purpose` labels the call in token metrics and lets tests fake each call type.
        """
        try:
            response = await self._client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                caller=self._caller_for(purpose),
                temperature=0,
                max_tokens=max_tokens,
                category=CATEGORY_FACTUAL_QA,
                response_format={"type": "json_object"},
            )
            _record(purpose, response.model, response.prompt_tokens, response.completion_tokens)
            return extract_json_object(response.content or "{}")
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM call '{purpose}' failed: {exc}") from exc

    async def complete_text(self, prompt: str, *, purpose: str, **values: str) -> str:
        """Plain completion from a registry prompt (e.g. chit-chat)."""
        system, user = self.prompt(prompt).render(organisation=self._organisation, **values)
        try:
            response = await self._client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                caller=self._caller_for(purpose),
                temperature=0.3,
                max_tokens=300,
                category=CATEGORY_CONVERSATIONAL,
            )
            _record(purpose, response.model, response.prompt_tokens, response.completion_tokens)
            return response.content or ""
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM call '{purpose}' failed: {exc}") from exc

    async def generate_answer(
        self,
        query: str,
        contexts: list[VectorSearchResult],
        history: History = (),
        prompt: str = "rag_answer",
    ) -> str:
        try:
            response = await self._client.chat(
                self._answer_messages(query, contexts, history, prompt),
                caller=self._caller_for("answer"),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                category=CATEGORY_FACTUAL_QA,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Chat completion failed: {exc}") from exc
        _record("answer", response.model, response.prompt_tokens, response.completion_tokens)
        if response.finish_reason == "length":
            logger.warning(
                "llm_answer_truncated", model=response.model, max_tokens=self._max_tokens
            )
        return response.content or ""

    async def stream_answer(
        self,
        query: str,
        contexts: list[VectorSearchResult],
        history: History = (),
        prompt: str = "rag_answer",
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they are generated (used by the SSE chat endpoint)."""
        try:
            async for event in self._client.stream(
                self._answer_messages(query, contexts, history, prompt),
                caller=self._caller_for("answer"),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                category=CATEGORY_FACTUAL_QA,
            ):
                if event.error:
                    raise GenerationError(f"Chat completion stream failed: {event.error}")
                if event.is_final:
                    _record("answer", event.model, event.prompt_tokens, event.completion_tokens)
                if event.content:
                    yield event.content
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Chat completion stream failed: {exc}") from exc

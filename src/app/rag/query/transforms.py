"""Query transformation library shared by all retrievers (C-163, C-164, S-23, S-24).

- multi_query: N paraphrases for RAG-Fusion (retrieve for each, then RRF)
- hyde:        a hypothetical answer passage, embedded instead of the question
- step_back:   a more general question that retrieves background context
- decompose:   ordered sub-questions for multi-step questions (also used by the planner)
Every transform returns the original question unchanged if the LLM call fails.
"""

from typing import Protocol

from app.core.exceptions import GenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)

MULTI_QUERY_PROMPT = (
    "Write {n} different search queries that would retrieve documents answering the user's "
    "question: vary wording, synonyms and specificity. Keep the language of the question. "
    'Return JSON: {{"queries": [str]}}.'
)
HYDE_PROMPT = (
    "Write a short passage (3-5 sentences) that would plausibly appear in an internal "
    "document answering the question. It is used only to search for real documents, so "
    'invent plausible details freely. Return JSON: {"passage": str}.'
)
STEP_BACK_PROMPT = (
    "Rewrite the question as a more general 'step-back' question about the underlying concept "
    'or policy, useful to retrieve background context. Return JSON: {"question": str}.'
)
DECOMPOSE_PROMPT = (
    "Break the question into the minimal ordered list of simpler sub-questions whose answers "
    "together answer it (at most {n}). If it is already simple, return it alone. "
    'Return JSON: {{"sub_questions": [str]}}.'
)


class JsonLLM(Protocol):
    async def complete_json(self, system: str, user: str, *, purpose: str) -> dict: ...


class QueryTransformer:
    def __init__(self, llm: JsonLLM) -> None:
        self._llm = llm

    async def _ask(self, prompt: str, question: str, purpose: str) -> dict:
        try:
            return await self._llm.complete_json(prompt, question, purpose=purpose)
        except GenerationError as exc:
            logger.warning("query_transform_failed", purpose=purpose, error=exc.message)
            return {}

    async def multi_query(self, question: str, n: int = 3) -> list[str]:
        data = await self._ask(MULTI_QUERY_PROMPT.format(n=n), question, "query_multi")
        queries = [str(q).strip() for q in data.get("queries") or [] if str(q).strip()]
        return list(dict.fromkeys([question, *queries]))[: n + 1]

    async def hyde(self, question: str) -> str:
        data = await self._ask(HYDE_PROMPT, question, "query_hyde")
        return str(data.get("passage") or question)

    async def step_back(self, question: str) -> str:
        data = await self._ask(STEP_BACK_PROMPT, question, "query_step_back")
        return str(data.get("question") or question)

    async def decompose(self, question: str, max_parts: int = 4) -> list[str]:
        data = await self._ask(DECOMPOSE_PROMPT.format(n=max_parts), question, "query_decompose")
        parts = [str(q).strip() for q in data.get("sub_questions") or [] if str(q).strip()]
        return parts[:max_parts] or [question]

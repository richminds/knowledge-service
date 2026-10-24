"""Answer cache (C-165, C-222).

Two tiers, both keyed by the *access scope* (RetrievalFilters.signature(): collection,
caller principals, metadata filters) plus RAG mode and prompt version, so a cached answer
is never served to someone who could not see its sources:
  - semantic (in-process): cosine similarity of question embeddings >= threshold
  - exact (Redis, optional): normalised question text; shared across instances
    (placeholder REDIS_URL)
Entries for a collection are invalidated when it is re-ingested.
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", question.lower())).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class _Entry:
    embedding: list[float] | None
    question: str
    payload: dict
    expires: float


class AnswerCache:
    def __init__(
        self,
        threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries_per_scope: int = 500,
        redis_url: str = "",
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_entries_per_scope
        self._scopes: dict[str, list[_Entry]] = {}
        self._redis = None
        if redis_url:
            from redis.asyncio import Redis

            self._redis = Redis.from_url(redis_url)

    @staticmethod
    def scope(
        filters_signature: str, rag_mode: str, prompt_version: str, vector_store: str = ""
    ) -> str:
        return f"{filters_signature}\x1e{rag_mode}\x1e{prompt_version}\x1e{vector_store}"

    @staticmethod
    def _redis_key(scope: str, question: str) -> str:
        collection = scope.split("\x1f", 1)[0]
        digest = hashlib.sha256(f"{scope}\x1d{normalise(question)}".encode()).hexdigest()
        return f"answercache:{collection}:{digest}"

    async def lookup(self, scope: str, question: str, embedding: list[float] | None) -> dict | None:
        now = time.time()
        entries = [e for e in self._scopes.get(scope, []) if e.expires > now]
        self._scopes[scope] = entries
        target = normalise(question)
        for entry in entries:
            if entry.question == target:
                return entry.payload
        if embedding is not None:
            best = max(
                ((e, _cosine(embedding, e.embedding)) for e in entries if e.embedding),
                key=lambda pair: pair[1],
                default=(None, 0.0),
            )
            if best[0] is not None and best[1] >= self._threshold:
                return best[0].payload
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._redis_key(scope, question))
                return json.loads(raw) if raw else None
            except Exception as exc:  # noqa: BLE001 - the cache must never break answering
                logger.warning("cache_unavailable", error=str(exc))
        return None

    async def store(
        self, scope: str, question: str, embedding: list[float] | None, payload: dict
    ) -> None:
        entries = self._scopes.setdefault(scope, [])
        entries.append(_Entry(embedding, normalise(question), payload, time.time() + self._ttl))
        del entries[: max(0, len(entries) - self._max)]
        if self._redis is not None:
            try:
                await self._redis.set(
                    self._redis_key(scope, question), json.dumps(payload, default=str), ex=self._ttl
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache_unavailable", error=str(exc))

    async def invalidate(self, collection: str) -> None:
        for scope in [s for s in self._scopes if s.split("\x1f", 1)[0] == collection]:
            del self._scopes[scope]
        if self._redis is not None:
            try:
                async for key in self._redis.scan_iter(match=f"answercache:{collection}:*"):
                    await self._redis.delete(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache_unavailable", error=str(exc))

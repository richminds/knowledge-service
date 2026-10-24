from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.azure_auth import build_openai_client
from app.core.exceptions import EmbeddingError
from app.core.usage import record_usage

# Only transient failures are retried; 4xx errors (bad input, auth) fail fast.
_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


class EmbeddingClient:
    """Thin async wrapper around Azure OpenAI embeddings with batching and retries."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        batch_size: int = 64,
        dimensions: int | None = None,
    ) -> None:
        if not endpoint:
            raise EmbeddingError("Azure OpenAI endpoint is not configured")
        self._client = build_openai_client(endpoint, api_key, api_version)
        self._deployment = deployment
        self._batch_size = batch_size
        self._dimensions = dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed any number of texts, splitting into API-sized batches."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            try:
                batch = await self._embed_once(texts[start : start + self._batch_size])
            except _TRANSIENT_ERRORS as exc:
                raise EmbeddingError(f"Embedding request failed after retries: {exc}") from exc
            embeddings.extend(batch)
        return embeddings

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _embed_once(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs = {"dimensions": self._dimensions} if self._dimensions else {}
        try:
            response = await self._client.embeddings.create(
                model=self._deployment, input=texts, **kwargs
            )
        except _TRANSIENT_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        if response.usage is not None:
            record_usage("embedding", self._deployment, response.usage.prompt_tokens or 0, 0)
        return [item.embedding for item in response.data]


class EmbeddingCache:
    """Embedding cache by content hash (C-127): unchanged text is never re-embedded.

    In-process LRU, plus Redis (REDIS_URL) so all instances/workers share it. Keys include
    the model and dimensions, so changing the embedding model never serves stale vectors
    (C-130). Azure OpenAI embeddings are already unit-normalised (C-128)."""

    def __init__(
        self, inner, model_key: str, max_entries: int = 50_000, redis_url: str = ""
    ) -> None:
        from collections import OrderedDict

        self._inner = inner
        self._model_key = model_key
        self._memory: OrderedDict[str, list[float]] = OrderedDict()
        self._max = max_entries
        self._redis = None
        if redis_url:
            from redis.asyncio import Redis

            self._redis = Redis.from_url(redis_url)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        import hashlib

        return "emb:" + hashlib.sha256(f"{self._model_key}\x1f{text}".encode()).hexdigest()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import json

        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {k: self._memory[k] for k in keys if k in self._memory}
        missing = [k for k in dict.fromkeys(keys) if k not in found]
        if missing and self._redis is not None:
            try:
                for key, raw in zip(missing, await self._redis.mget(missing), strict=True):
                    if raw:
                        found[key] = json.loads(raw)
            except Exception:  # noqa: BLE001 - cache must never break ingestion
                pass
        to_embed = [
            (k, t) for k, t in dict(zip(keys, texts, strict=True)).items() if k not in found
        ]
        if to_embed:
            vectors = await self._inner.embed_batch([t for _, t in to_embed])
            for (key, _), vector in zip(to_embed, vectors, strict=True):
                found[key] = vector
                if self._redis is not None:
                    try:
                        await self._redis.set(key, json.dumps(vector), ex=30 * 86400)
                    except Exception:  # noqa: BLE001
                        pass
        for key in keys:
            self._memory[key] = found[key]
            self._memory.move_to_end(key)
        while len(self._memory) > self._max:
            self._memory.popitem(last=False)
        self.hits += len(keys) - len(to_embed)
        self.misses += len(to_embed)
        return [found[k] for k in keys]

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

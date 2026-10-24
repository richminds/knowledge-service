import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.exceptions import EmbeddingError
from app.rag.generation.llm_gateway_sdk import GatewayError, RemoteLLMClient


def _transient(exc: BaseException) -> bool:
    """Retry only what may succeed next time: network errors, throttling, gateway 5xx.
    4xx errors (bad input, auth) fail fast."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, GatewayError) and (exc.status_code == 429 or exc.status_code >= 500)


class EmbeddingClient:
    """Embeddings through the LLM gateway (POST /v1/embeddings) with batching and retries.

    The gateway owns the embedding model (LLM_EMBEDDING_* on the gateway); `dimensions`
    must match it, because the Atlas vector index is created with that size. A mismatch
    fails loudly here instead of silently corrupting the index."""

    def __init__(
        self,
        client: RemoteLLMClient,
        batch_size: int = 64,
        dimensions: int | None = None,
        caller: str = "knowledge-service",
    ) -> None:
        self._client = client
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._caller = f"{caller}:embedding"

    async def embed_batch(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Embed any number of texts, splitting into API-sized batches."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            try:
                batch = await self._embed_once(texts[start : start + self._batch_size], input_type)
            except (httpx.HTTPError, GatewayError) as exc:
                raise EmbeddingError(f"Embedding request failed after retries: {exc}") from exc
            embeddings.extend(batch)
        return embeddings

    async def embed_text(self, text: str) -> list[float]:
        """A single search string (embedded as a query for asymmetric models)."""
        embeddings = await self.embed_batch([text], input_type="query")
        return embeddings[0]

    @retry(
        retry=retry_if_exception(_transient),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _embed_once(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._client.embed(texts, input_type=input_type, caller=self._caller)
        if self._dimensions and vectors and len(vectors[0]) != self._dimensions:
            raise EmbeddingError(
                f"The gateway's embedding model returns {len(vectors[0])} dimensions but "
                f"EMBEDDING_DIMENSIONS={self._dimensions}; set them to match"
            )
        return vectors


class EmbeddingCache:
    """Embedding cache by content hash (C-127): unchanged text is never re-embedded.

    In-process LRU, plus Redis (REDIS_URL) so all instances/workers share it. Keys include
    the model and dimensions, so changing the embedding model never serves stale vectors
    (C-130)."""

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

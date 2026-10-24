"""Connector contract (C-057) and shared HTTP behaviour (C-065)."""

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from app.core.logging import get_logger
from app.rag.ingestion.models import SourceDocument

logger = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class Connector(ABC):
    """A data source. `iterate` yields every document under a URI with its ACL and change
    markers; the pipeline handles type detection, parsing and change detection."""

    #: URI prefixes this connector handles, e.g. ("sharepoint://",)
    schemes: tuple[str, ...] = ()

    @abstractmethod
    def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        """Yield the documents under `uri`."""

    def source_prefix(self, uri: str) -> str:
        """Prefix shared by the `source` of every document under `uri` (full-sync deletes)."""
        return uri

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release clients."""


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
) -> httpx.Response:
    """HTTP call with exponential backoff + full jitter; honours Retry-After (C-065)."""
    for attempt in range(1, attempts + 1):
        try:
            response = await client.request(method, url, **kwargs)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt == attempts:
                raise
            wait = random.uniform(0, min(max_delay, base_delay * 2**attempt))  # noqa: S311
            logger.info("http_retry", url=url, attempt=attempt, error=type(exc).__name__)
            await asyncio.sleep(wait)
            continue
        if response.status_code not in _RETRY_STATUS or attempt == attempts:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After", "")
        wait = (
            float(retry_after)
            if retry_after.replace(".", "", 1).isdigit()
            else random.uniform(0, min(max_delay, base_delay * 2**attempt))  # noqa: S311
        )
        logger.info("http_retry", url=url, attempt=attempt, status=response.status_code)
        await asyncio.sleep(min(wait, max_delay))
    raise RuntimeError("unreachable")

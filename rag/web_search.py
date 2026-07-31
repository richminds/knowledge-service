"""Free web search retrieval leg for RAG augmentation (Tavily).

Runs as a fourth retrieval source alongside semantic/keyword/graph search
(see retrieval.py's ``_web_search`` node). Disabled by default via
``RAG_WEB_SEARCH_ENABLED``.

This store does NOT return a fabricated stub when ``TAVILY_API_KEY`` is
absent — a fake "search result" would be presented to the LLM as real
retrieved evidence and could be cited in the answer. Silence (empty list) is
the only safe degradation for a retrieval source. Real HTTP failures are
caught for the same reason and also degrade to an empty list rather than
raising, so a flaky web search never fails the whole query.

Web results are fetched fresh per query (never persisted to the vector
store) and flow through the same fusion -> parent-expand -> rerank ->
knee-point -> prompt-injection safety filter pipeline as internal chunks
before reaching the LLM — see retrieval.py's ``_safety_filter`` node, which
runs on the full merged ``selected_chunks`` list regardless of source.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.documents import Document

from .vector_store import ScoredDocument

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
_TAVILY_URL = "https://api.tavily.com/search"


class WebSearchStore:
    """Tavily-backed free web search adapter (free tier, no cost)."""

    name = "tavily"

    def __init__(self) -> None:
        self._api_key = os.getenv("TAVILY_API_KEY", "")

    async def retrieve(self, query: str, top_k: int = 3) -> list[ScoredDocument]:
        """Return up to ``top_k`` web results for ``query``, or [] if unavailable."""
        if not self._api_key:
            logger.info("WebSearchStore: TAVILY_API_KEY not set — skipping web search.")
            return []

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _TAVILY_URL,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": top_k,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
            if resp.status_code not in range(200, 300):
                logger.warning(
                    "WebSearchStore: Tavily search failed: HTTP %s — %s",
                    resp.status_code, resp.text[:200],
                )
                return []
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WebSearchStore: Tavily search unavailable (%s: %s)", type(exc).__name__, exc
            )
            return []

        return [
            ScoredDocument(
                document=Document(
                    page_content=str(item.get("content", "")),
                    metadata=_result_metadata(item, idx),
                ),
                score=float(item.get("score", 0.0)),
                source="web",
            )
            for idx, item in enumerate(payload.get("results", [])[:top_k], start=1)
            if item.get("content")
        ]


def _result_metadata(item: dict[str, Any], idx: int) -> dict[str, Any]:
    url = str(item.get("url", ""))
    return {
        "source": url,
        "title": str(item.get("title") or "untitled"),
        "section": "web",
        "page_number": "n/a",
        "chunk_id": f"web-{idx}-{abs(hash(url)) & 0xFFFFFFFF:08x}",
        "chunk_index": idx,
        "published_date": item.get("published_date", ""),
    }

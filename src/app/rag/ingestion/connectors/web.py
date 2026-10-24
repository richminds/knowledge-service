"""Polite web crawler (C-063): same-host links only, robots.txt, depth/page/rate limits.

URI: https://intranet.example.org/policies/ (crawls pages and linked PDFs/Office files
under the same host). Limits: WEB_CRAWL_MAX_DEPTH, WEB_CRAWL_MAX_PAGES,
WEB_CRAWL_DELAY_SECONDS. Pages behind sign-in need a header (WEB_CRAWL_AUTH_HEADER).
"""

import asyncio
import re
from collections.abc import AsyncIterator
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from app.core.logging import get_logger
from app.rag.ingestion.connectors.base import Connector
from app.rag.ingestion.models import SourceDocument

logger = get_logger(__name__)

_HREF = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.IGNORECASE)
_DOCUMENT_EXT = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md")
_USER_AGENT = "rag-knowledge-crawler/1.0"


class WebConnector(Connector):
    schemes = ("https://", "http://")

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 200,
        delay_seconds: float = 1.0,
        auth_header: str = "",
        acl: list[str] | None = None,
    ) -> None:
        headers = {"User-Agent": _USER_AGENT}
        if auth_header:
            name, _, value = auth_header.partition(":")
            headers[name.strip()] = value.strip()
        self._http = httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True)
        self._max_depth = max_depth
        self._max_pages = max_pages
        self._delay = delay_seconds
        self._acl = acl or []

    async def _robots(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        robots = RobotFileParser()
        try:
            response = await self._http.get(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
            robots.parse(response.text.splitlines() if response.status_code == 200 else [])
        except httpx.HTTPError:
            robots.parse([])
        return robots

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        start = urldefrag(uri)[0]
        host = urlparse(start).netloc
        robots = await self._robots(start)
        queue: list[tuple[str, int]] = [(start, 0)]
        seen: set[str] = {start}
        fetched = 0
        while queue and fetched < self._max_pages:
            url, depth = queue.pop(0)
            if not robots.can_fetch(_USER_AGENT, url):
                continue
            try:
                response = await self._http.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.info("crawl_skip", url=url, error=str(exc))
                continue
            fetched += 1
            content_type = response.headers.get("content-type", "")
            yield SourceDocument(
                source=url,
                name=urlparse(url).path.rsplit("/", 1)[-1] or "index.html",
                content=response.content,
                content_type=content_type,
                acl=list(self._acl),
                etag=response.headers.get("etag", ""),
                last_modified=response.headers.get("last-modified"),
                url=url,
            )
            if "html" in content_type and depth < self._max_depth:
                for href in _HREF.findall(response.text):
                    link = urldefrag(urljoin(url, href))[0]
                    parsed = urlparse(link)
                    if (
                        parsed.netloc != host
                        or link in seen
                        or parsed.scheme not in ("http", "https")
                    ):
                        continue
                    path = parsed.path.lower()
                    if "." in path.rsplit("/", 1)[-1] and not path.endswith(
                        _DOCUMENT_EXT + (".html", ".htm")
                    ):
                        continue
                    seen.add(link)
                    queue.append((link, depth + 1))
            await asyncio.sleep(self._delay)

    async def close(self) -> None:
        await self._http.aclose()

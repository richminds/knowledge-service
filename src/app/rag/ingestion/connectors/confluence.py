"""Confluence (Cloud or Data Center) connector (C-060).

URI: confluence://<SPACEKEY>  (all pages of a space, with hierarchy and attachments)
Placeholders: CONFLUENCE_BASE_URL (https://<org>.atlassian.net/wiki), CONFLUENCE_USER and
CONFLUENCE_API_TOKEN (Key Vault: confluence-api-token).

Permissions: page read restrictions (groups/users) become the ACL. Confluence group names
are not Entra ids: map them in config/acl_mapping.yaml (TODO); unmapped restricted pages
get an empty ACL and are therefore invisible (fail closed). Unrestricted pages inherit
the space permissions -> CONFLUENCE_DEFAULT_ACL (default "*").
"""

from collections.abc import AsyncIterator

import httpx

from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.connectors.base import Connector, request_with_retry
from app.rag.ingestion.models import SourceDocument

_EXPAND = (
    "body.storage,version,ancestors,history,"
    "restrictions.read.restrictions.group,restrictions.read.restrictions.user"
)


class ConfluenceConnector(Connector):
    schemes = ("confluence://",)

    def __init__(
        self,
        base_url: str,
        user: str,
        api_token: str,
        group_map: dict[str, str] | None = None,
        default_acl: list[str] | None = None,
        include_attachments: bool = True,
    ) -> None:
        if not base_url or not api_token:
            raise DocumentLoadError("Confluence connector is not configured")
        self._base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(auth=(user, api_token), timeout=60.0)
        self._group_map = group_map or {}
        self._default_acl = default_acl or ["*"]
        self._attachments = include_attachments

    def _acl(self, page: dict) -> list[str]:
        restrictions = (
            ((page.get("restrictions") or {}).get("read") or {}).get("restrictions")
        ) or {}
        groups = [g.get("name", "") for g in (restrictions.get("group") or {}).get("results", [])]
        users = [
            u.get("accountId", "") for u in (restrictions.get("user") or {}).get("results", [])
        ]
        if not groups and not users:
            return list(self._default_acl)
        return sorted({self._group_map[g] for g in groups if g in self._group_map})

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        space = uri.removeprefix("confluence://").strip("/")
        url = f"{self._base}/rest/api/content"
        params = {"spaceKey": space, "type": "page", "expand": _EXPAND, "limit": 50, "start": 0}
        while True:
            page_batch = (await request_with_retry(self._http, "GET", url, params=params)).json()
            for page in page_batch.get("results", []):
                acl = self._acl(page)
                breadcrumb = [a.get("title", "") for a in page.get("ancestors", [])]
                html = (page.get("body") or {}).get("storage", {}).get("value", "")
                link = self._base + ((page.get("_links") or {}).get("webui") or "")
                yield SourceDocument(
                    source=f"confluence://{space}/{page['id']}",
                    name=f"{page.get('title', page['id'])}.html",
                    content=(
                        f"<html><body><h1>{page.get('title', '')}</h1>{html}</body></html>"
                    ).encode(),
                    content_type="text/html",
                    acl=acl,
                    etag=str((page.get("version") or {}).get("number", "")),
                    last_modified=(page.get("version") or {}).get("when"),
                    url=link,
                    author=((page.get("history") or {}).get("createdBy") or {}).get(
                        "displayName", ""
                    ),
                    metadata={"space": space, "breadcrumb": " > ".join(breadcrumb)},
                )
                if self._attachments:
                    async for attachment in self._page_attachments(space, page, acl, link):
                        yield attachment
            if page_batch.get("size", 0) < params["limit"]:
                return
            params["start"] += params["limit"]

    async def _page_attachments(self, space, page, acl, link) -> AsyncIterator[SourceDocument]:
        url = f"{self._base}/rest/api/content/{page['id']}/child/attachment"
        listing = (await request_with_retry(self._http, "GET", url, params={"limit": 100})).json()
        for item in listing.get("results", []):
            download = self._base + ((item.get("_links") or {}).get("download") or "")
            content = (
                await request_with_retry(self._http, "GET", download, follow_redirects=True)
            ).content
            yield SourceDocument(
                source=f"confluence://{space}/{page['id']}/attachments/{item['id']}",
                name=item.get("title", item["id"]),
                content=content,
                content_type=(item.get("metadata") or {}).get("mediaType", ""),
                acl=acl,
                etag=str((item.get("version") or {}).get("number", "")),
                url=link,
                metadata={"space": space, "parent_page": page.get("title", "")},
            )

    async def close(self) -> None:
        await self._http.aclose()

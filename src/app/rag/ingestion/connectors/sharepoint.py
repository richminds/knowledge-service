"""SharePoint Online / OneDrive connector via Microsoft Graph (C-059, C-064, C-066).

URI: sharepoint://<hostname>/sites/<site>/<library>[/<folder path>]
     e.g. sharepoint://contoso.sharepoint.com/sites/HR/Shared Documents/Policies

Auth (placeholders): SHAREPOINT_TENANT_ID + SHAREPOINT_CLIENT_ID + SHAREPOINT_CLIENT_SECRET
(Key Vault: sharepoint-client-secret) for an app registration with Sites.Selected
(granted per site) or Sites.Read.All; or leave the secret blank to use the managed
identity (DefaultAzureCredential) with the same Graph application permission.

Permissions: item permissions become the document ACL - Entra group / user object ids;
organisation-wide sharing links map to "*". SharePoint site groups have no Entra id;
map them in config/acl_mapping.yaml (TODO) or they are skipped (fail closed).
"""

from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from app.core.exceptions import DocumentLoadError
from app.core.logging import get_logger
from app.rag.ingestion.connectors.base import Connector, request_with_retry
from app.rag.ingestion.models import SourceDocument

logger = get_logger(__name__)

GRAPH = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"


def acl_from_permissions(permissions: list[dict], site_group_map: dict[str, str]) -> list[str]:
    acl: set[str] = set()
    for perm in permissions:
        link = perm.get("link") or {}
        if link.get("scope") in ("organization", "anonymous"):
            acl.add("*")
        for grantee in [perm.get("grantedToV2") or {}, *(perm.get("grantedToIdentitiesV2") or [])]:
            for kind in ("group", "user"):
                if principal := (grantee.get(kind) or {}).get("id"):
                    acl.add(principal)
            if site_group := (grantee.get("siteGroup") or {}).get("displayName"):
                if mapped := site_group_map.get(site_group):
                    acl.add(mapped)
    return sorted(acl)


class SharePointConnector(Connector):
    schemes = ("sharepoint://",)

    def __init__(
        self,
        tenant_id: str = "",
        client_id: str = "",
        client_secret: str = "",
        credential=None,
        site_group_map: dict[str, str] | None = None,
    ) -> None:
        if client_secret:
            from azure.identity.aio import ClientSecretCredential

            self._credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        else:
            self._credential = credential
        if self._credential is None:
            raise DocumentLoadError("SharePoint connector is not configured")
        self._site_group_map = site_group_map or {}
        self._http = httpx.AsyncClient(timeout=60.0)

    async def _get(self, url: str) -> dict:
        token = await self._credential.get_token(GRAPH_SCOPE)
        response = await request_with_retry(
            self._http, "GET", url, headers={"Authorization": f"Bearer {token.token}"}
        )
        return response.json()

    async def _download(self, url: str) -> bytes:
        token = await self._credential.get_token(GRAPH_SCOPE)
        response = await request_with_retry(
            self._http,
            "GET",
            url,
            headers={"Authorization": f"Bearer {token.token}"},
            follow_redirects=True,
        )
        return response.content

    async def _children(self, drive_id: str, item_path: str) -> AsyncIterator[dict]:
        url = (
            f"{GRAPH}/drives/{drive_id}/root:/{quote(item_path)}:/children"
            if item_path
            else f"{GRAPH}/drives/{drive_id}/root/children"
        )
        while url:
            page = await self._get(url)
            for item in page.get("value", []):
                yield item
            url = page.get("@odata.nextLink", "")

    async def _walk(self, drive_id: str, path: str) -> AsyncIterator[dict]:
        async for item in self._children(drive_id, path):
            child = f"{path}/{item['name']}".strip("/")
            if "folder" in item:
                async for nested in self._walk(drive_id, child):
                    yield nested
            elif "file" in item:
                yield item

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        rest = uri.removeprefix("sharepoint://")
        hostname, _, path = rest.partition("/")
        parts = path.split("/")
        if len(parts) < 3 or parts[0] != "sites":
            raise DocumentLoadError("Use sharepoint://<host>/sites/<site>/<library>[/<folder>]")
        site_path, library, folder = parts[1], parts[2], "/".join(parts[3:])
        site = await self._get(f"{GRAPH}/sites/{hostname}:/sites/{site_path}")
        drives = await self._get(f"{GRAPH}/sites/{site['id']}/drives")
        drive = next((d for d in drives.get("value", []) if d.get("name") == library), None)
        if drive is None:
            raise DocumentLoadError(f"Library {library!r} not found in site {site_path!r}")

        async for item in self._walk(drive["id"], folder):
            permissions = await self._get(
                f"{GRAPH}/drives/{drive['id']}/items/{item['id']}/permissions"
            )
            yield SourceDocument(
                source=f"sharepoint://{hostname}/sites/{site_path}/{library}/{item['id']}",
                name=item["name"],
                content=await self._download(
                    f"{GRAPH}/drives/{drive['id']}/items/{item['id']}/content"
                ),
                content_type=(item.get("file") or {}).get("mimeType", ""),
                acl=acl_from_permissions(permissions.get("value", []), self._site_group_map),
                etag=item.get("eTag", ""),
                last_modified=item.get("lastModifiedDateTime"),
                url=item.get("webUrl", ""),
                author=((item.get("createdBy") or {}).get("user") or {}).get("displayName", ""),
                sensitivity_label=(item.get("sensitivityLabel") or {}).get("displayName", ""),
                metadata={"site": site_path, "library": library},
            )

    async def close(self) -> None:
        await self._http.aclose()

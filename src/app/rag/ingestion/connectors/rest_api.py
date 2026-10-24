"""Generic paginated REST connector (C-062) for Jira, ServiceNow, CRM knowledge bases...

URI: api://<name>, defined under `apis:` in config/sources.yaml:
    url: https://<instance>.service-now.com/api/now/table/kb_knowledge
    headers: {Authorization: "Bearer ${SERVICENOW_TOKEN}"}
    params: {sysparm_query: "workflow_state=published"}
    items_path: result                 # dot path to the list of items
    pagination: {type: offset, param: sysparm_offset, size_param: sysparm_limit, page_size: 100}
                # or {type: next_link, path: _links.next}  or {type: none}
    id_field: sys_id
    title_field: short_description
    text_fields: [short_description, text]
    url_template: https://<instance>.service-now.com/kb_view.do?sys_kb_id={sys_id}
    updated_field: sys_updated_on
    acl: ["*"]
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.rag.ingestion.connectors.base import Connector, request_with_retry
from app.rag.ingestion.connectors.sources_config import source_definition
from app.rag.ingestion.models import SourceDocument


def dot_get(data: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        data = data.get(part) if isinstance(data, dict) else None
    return data


class RestApiConnector(Connector):
    schemes = ("api://",)

    def __init__(self, sources_path: str) -> None:
        self._sources_path = sources_path
        self._http = httpx.AsyncClient(timeout=60.0)

    async def _pages(self, definition: dict) -> AsyncIterator[list[dict]]:
        pagination = definition.get("pagination") or {"type": "none"}
        params = dict(definition.get("params") or {})
        url = definition["url"]
        if pagination["type"] == "offset":
            params[pagination.get("size_param", "limit")] = pagination.get("page_size", 100)
            params[pagination["param"]] = 0
        while url:
            body = (
                await request_with_retry(
                    self._http, "GET", url, params=params, headers=definition.get("headers") or {}
                )
            ).json()
            items = dot_get(body, definition.get("items_path", "")) or []
            yield items
            kind = pagination["type"]
            if kind == "offset" and len(items) == pagination.get("page_size", 100):
                params[pagination["param"]] += len(items)
            elif kind == "next_link" and (link := dot_get(body, pagination["path"])):
                url, params = link, {}
            else:
                url = ""

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        name = uri.removeprefix("api://").strip("/")
        definition = source_definition(self._sources_path, "apis", name)
        async for items in self._pages(definition):
            for item in items:
                values = {
                    k: "" if v is None else str(v)
                    for k, v in item.items()
                    if not isinstance(v, (dict, list))
                }
                item_id = values.get(definition.get("id_field", "id"), "")
                title = values.get(definition.get("title_field", ""), "")
                text = "\n\n".join(
                    values.get(f, "")
                    for f in definition.get("text_fields") or values
                    if values.get(f)
                )
                yield SourceDocument(
                    source=f"api://{name}/{item_id}",
                    name=f"{name}-{item_id}.html" if "<" in text else f"{name}-{item_id}.txt",
                    content=(
                        f"<h1>{title}</h1>{text}" if "<" in text else f"# {title}\n\n{text}"
                    ).encode(),
                    acl=list(definition.get("acl") or []),
                    etag=values.get(definition.get("updated_field", ""), ""),
                    last_modified=values.get(definition.get("updated_field", "")) or None,
                    url=(definition.get("url_template") or "").format_map(values),
                    metadata={"title": title, "api_source": name},
                )

    async def close(self) -> None:
        await self._http.aclose()

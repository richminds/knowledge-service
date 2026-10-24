"""Azure AI Document Intelligence layout parser (C-074..C-078, C-085).

prebuilt-layout gives OCR for scanned pages (C-075), reading order and roles (title,
sectionHeading, pageHeader/pageFooter/pageNumber -> dropped, C-090), tables (C-077) with
multi-page tables stitched, figures with captions (C-078) and page numbers / bounding
boxes for every element (C-085).

Placeholders: DOC_INTELLIGENCE_ENDPOINT (https://<name>.cognitiveservices.azure.com) and
DOC_INTELLIGENCE_API_KEY (blank = managed identity). Used for PDFs (PDF_PARSER=docintel|auto,
always prebuilt-layout) and images (IMAGE_PARSER, model DOC_INTELLIGENCE_IMAGE_MODEL:
prebuilt-layout or the cheaper OCR-only prebuilt-read). Supported image formats: PNG, JPEG,
BMP, TIFF, HEIF (see filetypes.IMAGE_FORMATS); the service limit is 500 MB / 2,000 pages (S0).
"""

import asyncio

import httpx

from app.core.azure_auth import COGNITIVE_SERVICES_SCOPE, get_azure_credential
from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.models import Element, ParsedDocument

_API_VERSION = "2024-11-30"
_DROP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


def _table_markdown(table: dict) -> str:
    rows: dict[int, dict[int, str]] = {}
    for cell in table.get("cells", []):
        rows.setdefault(cell["rowIndex"], {})[cell["columnIndex"]] = (
            cell.get("content", "").replace("|", "\\|").replace("\n", " ")
        )
    columns = table.get("columnCount", 0)
    lines = []
    for r in sorted(rows):
        lines.append("| " + " | ".join(rows[r].get(c, "") for c in range(columns)) + " |")
        if r == 0:
            lines.append("|" + " --- |" * columns)
    return "\n".join(lines)


def _page(item: dict) -> int | None:
    regions = item.get("boundingRegions") or []
    return regions[0].get("pageNumber") if regions else None


def _bbox(item: dict) -> tuple[float, ...] | None:
    regions = item.get("boundingRegions") or []
    return tuple(regions[0].get("polygon") or ()) or None if regions else None


def layout_to_document(
    source: str, title: str, result: dict, model: str = "prebuilt-layout"
) -> ParsedDocument:
    """Map an analyzeResult (layout) to ordered elements with a section tree."""
    tables = result.get("tables") or []
    table_spans = [
        (span["offset"], span["offset"] + span["length"])
        for t in tables
        for span in t.get("spans", [])
    ]

    def in_table(offset: int) -> bool:
        return any(start <= offset < end for start, end in table_spans)

    items: list[tuple[int, str, dict]] = []
    for paragraph in result.get("paragraphs") or []:
        offset = (paragraph.get("spans") or [{"offset": 0}])[0]["offset"]
        if paragraph.get("role") in _DROP_ROLES or in_table(offset):
            continue
        items.append((offset, "paragraph", paragraph))
    # C-077: a table continuing on the next page with the same columns is one table.
    merged: list[dict] = []
    for table in tables:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.get("columnCount") == table.get("columnCount")
            and (_page(table) or 0) == (_page(previous) or 0) + 1
        ):
            offset_rows = previous.get("rowCount", 0)
            previous["cells"] = previous["cells"] + [
                {**c, "rowIndex": c["rowIndex"] + offset_rows}
                for c in table.get("cells", [])
                if c.get("kind") != "columnHeader"
            ]
            previous["rowCount"] = offset_rows + table.get("rowCount", 0)
            continue
        merged.append(dict(table))
    for table in merged:
        offset = (table.get("spans") or [{"offset": 0}])[0]["offset"]
        items.append((offset, "table", table))
    for figure in result.get("figures") or []:
        offset = (figure.get("spans") or [{"offset": 0}])[0]["offset"]
        items.append((offset, "figure", figure))
    items.sort(key=lambda item: item[0])

    elements: list[Element] = []
    section: list[str] = []
    for _, kind, item in items:
        if kind == "table":
            elements.append(
                Element(
                    "table",
                    _table_markdown(item),
                    page=_page(item),
                    section_path=list(section),
                    bbox=_bbox(item),
                )
            )
            continue
        if kind == "figure":
            caption = (item.get("caption") or {}).get("content", "")
            if caption:
                elements.append(
                    Element(
                        "figure",
                        f"Figure: {caption}",
                        page=_page(item),
                        section_path=list(section),
                        bbox=_bbox(item),
                    )
                )
            continue
        role, text = item.get("role"), item.get("content", "").strip()
        if not text:
            continue
        if role in ("title", "sectionHeading"):
            level = 1 if role == "title" else 2
            section = section[: level - 1] + [text]
            elements.append(
                Element(
                    "title" if role == "title" else "heading",
                    text,
                    level=level,
                    page=_page(item),
                    section_path=list(section),
                    bbox=_bbox(item),
                )
            )
        else:
            elements.append(
                Element(
                    "paragraph",
                    text,
                    page=_page(item),
                    section_path=list(section),
                    bbox=_bbox(item),
                )
            )
    first_title = next((e.text for e in elements if e.type == "title"), title)
    return ParsedDocument(
        source=source,
        title=first_title,
        elements=elements,
        parser=f"docintel:{model}",
        page_count=len(result.get("pages") or []) or None,
    )


class DocumentIntelligenceParser:
    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        timeout_seconds: int = 300,
        image_model: str = "prebuilt-layout",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._image_model = image_model
        self._timeout = timeout_seconds
        self._http = httpx.AsyncClient(timeout=60.0)

    async def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Ocp-Apim-Subscription-Key": self._api_key}
        token = await get_azure_credential().get_token(COGNITIVE_SERVICES_SCOPE)
        return {"Authorization": f"Bearer {token.token}"}

    async def analyze(self, content: bytes, model: str = "prebuilt-layout") -> dict:
        url = f"{self._endpoint}/documentintelligence/documentModels/{model}:analyze"
        headers = await self._headers()
        response = await self._http.post(
            url,
            params={"api-version": _API_VERSION, "features": "ocrHighResolution"},
            content=content,
            headers={**headers, "Content-Type": "application/octet-stream"},
        )
        if response.status_code != 202:
            raise DocumentLoadError(
                f"Document Intelligence rejected the file: {response.text[:200]}"
            )
        poll_url = response.headers["Operation-Location"]
        waited = 0.0
        while waited < self._timeout:
            await asyncio.sleep(2)
            waited += 2
            status = (await self._http.get(poll_url, headers=await self._headers())).json()
            if status.get("status") == "succeeded":
                return status["analyzeResult"]
            if status.get("status") == "failed":
                raise DocumentLoadError(f"Document Intelligence failed: {status.get('error')}")
        raise DocumentLoadError("Document Intelligence timed out")

    async def parse(self, source: str, name: str, content: bytes) -> ParsedDocument:
        return layout_to_document(source, name.rsplit(".", 1)[0], await self.analyze(content))

    async def parse_image(self, source: str, name: str, content: bytes) -> ParsedDocument:
        """OCR an image (scanned page, screenshot, photo of a form) with the image model."""
        result = await self.analyze(content, self._image_model)
        return layout_to_document(source, name.rsplit(".", 1)[0], result, self._image_model)

    async def close(self) -> None:
        await self._http.aclose()

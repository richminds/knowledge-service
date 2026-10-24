"""Azure Blob Storage / ADLS Gen2 connector (C-058): blob://<container>/<prefix>.

Per-blob metadata conventions (set by the uploader or a sync job):
  acl          comma-separated principal ids allowed to read the document (C-066)
  sensitivity  public | internal | confidential | restricted (C-095)
  url          human-viewable link for citations
Microsoft Defender for Storage malware scanning writes the blob index tag
"Malware Scanning scan result"; it is surfaced as metadata["malware_scan"] (C-071).
"""

from collections.abc import AsyncIterator

from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.connectors.base import Connector
from app.rag.ingestion.models import SourceDocument

BLOB_SCHEME = "blob://"
_SCAN_TAG = "Malware Scanning scan result"


class BlobConnector(Connector):
    schemes = (BLOB_SCHEME,)

    def __init__(self, connection_string: str = "", account_url: str = "", credential=None) -> None:
        self._connection_string = connection_string
        self._account_url = account_url
        self._credential = credential

    def _service(self):
        from azure.storage.blob.aio import BlobServiceClient

        if self._connection_string:
            return BlobServiceClient.from_connection_string(self._connection_string)
        if self._account_url:
            return BlobServiceClient(self._account_url, credential=self._credential)
        raise DocumentLoadError("Azure Storage is not configured")

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        container, _, prefix = uri.removeprefix(BLOB_SCHEME).partition("/")
        if not container:
            raise DocumentLoadError("Blob source must look like blob://<container>/<prefix>")
        async with self._service() as service:
            client = service.get_container_client(container)
            async for blob in client.list_blobs(
                name_starts_with=prefix or None, include=["metadata", "tags"]
            ):
                meta = {k.lower(): v for k, v in (blob.metadata or {}).items()}
                tags = blob.tags or {}
                downloader = await client.download_blob(blob.name)
                yield SourceDocument(
                    source=f"{BLOB_SCHEME}{container}/{blob.name}",
                    name=blob.name.rsplit("/", 1)[-1],
                    content=await downloader.readall(),
                    content_type=(
                        (blob.content_settings.content_type or "") if blob.content_settings else ""
                    ),
                    acl=[a.strip() for a in meta.get("acl", "").split(",") if a.strip()],
                    etag=blob.etag or "",
                    last_modified=blob.last_modified.isoformat() if blob.last_modified else None,
                    url=meta.get("url", ""),
                    sensitivity_label=meta.get("sensitivity", ""),
                    metadata={
                        "container": container,
                        **({"malware_scan": tags[_SCAN_TAG]} if _SCAN_TAG in tags else {}),
                    },
                )

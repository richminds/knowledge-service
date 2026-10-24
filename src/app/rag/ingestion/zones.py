"""Data-lake zones (C-030, C-068, C-099, C-142): every stage's output is persisted so any
stage can be re-run without going back to the source, and for audit.

  raw/        original bytes as received (C-068)
  parsed/     canonical markdown + elements JSON (C-086)
  clean/      cleaned text (silver, C-099)
  chunks/     chunk JSON as indexed
  quarantine/ rejected files + reason (C-070)

ZONES_STORE=local (ZONES_LOCAL_ROOT, default ./.ragdata) | blob (DATA_LAKE_ACCOUNT_URL +
DATA_LAKE_CONTAINER, managed identity) | off.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Protocol

ZONES = ("raw", "parsed", "clean", "chunks", "quarantine")


def _key(zone: str, collection: str, source: str, name: str) -> str:
    source_key = hashlib.sha256(source.encode()).hexdigest()[:24]
    return f"{zone}/{collection}/{source_key}/{name}"


class ZoneStore(Protocol):
    async def write(
        self, zone: str, collection: str, source: str, name: str, data: bytes | str
    ) -> str: ...


class NullZoneStore:
    async def write(self, zone, collection, source, name, data) -> str:
        return ""


class LocalZoneStore:
    def __init__(self, root: str) -> None:
        self._root = Path(root)

    async def write(
        self, zone: str, collection: str, source: str, name: str, data: bytes | str
    ) -> str:
        path = self._root / _key(zone, collection, source, name)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                path.write_text(data, encoding="utf-8")
            else:
                path.write_bytes(data)
            meta = path.parent / "_source.json"
            if not meta.exists():
                meta.write_text(
                    json.dumps({"source": source, "collection": collection}), encoding="utf-8"
                )

        await asyncio.to_thread(_write)
        return str(path)


class BlobZoneStore:
    def __init__(self, account_url: str, container: str, credential) -> None:
        from azure.storage.blob.aio import BlobServiceClient

        self._service = BlobServiceClient(account_url, credential=credential)
        self._container = container

    async def write(
        self, zone: str, collection: str, source: str, name: str, data: bytes | str
    ) -> str:
        key = _key(zone, collection, source, name)
        blob = self._service.get_blob_client(self._container, key)
        payload = data.encode() if isinstance(data, str) else data
        await blob.upload_blob(
            payload, overwrite=True, metadata={"source": source[:1024], "collection": collection}
        )
        return f"blob://{self._container}/{key}"

    async def close(self) -> None:
        await self._service.close()

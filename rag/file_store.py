"""Durable storage for raw uploaded knowledge files — MongoDB GridFS.

The bytes of every file uploaded through ``POST /v1/upload`` are persisted here
so they survive process/container restarts and can be re-ingested or audited.

MongoDB GridFS is the sole backend — no external blob store (S3, etc.) is used
or configured anywhere in this service. GridFS lives in the same shared
database (``RAG_MONGO_DB_NAME``) as every other collection, chunked
automatically for files above the 16 MB BSON limit. ``get_file_store()``
returns the store instance; the :class:`FileStore` Protocol exists so callers
depend on the interface, not the concrete class.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

class StoredFile(BaseModel):
    """Reference to a persisted file, returned by every FileStore backend."""

    file_id: str
    filename: str
    size: int
    backend: str
    content_type: str | None = None
    uploaded_at: str | None = None
    # End-user this file was uploaded on behalf of (see rag/loader.py's
    # `uploaded_by` document metadata for the same concept applied to
    # ingested chunks) — provenance only, not an access-control field.
    uploaded_by: str | None = None


@runtime_checkable
class FileStore(Protocol):
    """Backend-agnostic interface for persisting raw uploaded files."""

    backend: str

    async def save(
        self,
        filename: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredFile: ...

    async def read(self, file_id: str) -> bytes: ...

    async def delete(self, file_id: str) -> bool: ...

    async def list(self, limit: int = 100) -> list[StoredFile]: ...


# ---------------------------------------------------------------------------
# MongoDB GridFS implementation (default)
# ---------------------------------------------------------------------------

class MongoGridFSFileStore:
    """Persist file bytes in MongoDB via GridFS.

    GridFS chunks large files automatically, so this handles documents above
    the 16 MB BSON limit. Files live in ``<bucket>.files`` / ``<bucket>.chunks``
    in the shared database.
    """

    backend = "mongodb"

    def __init__(self) -> None:
        self._bucket_name = settings.file_store_gridfs_bucket

    async def _bucket(self) -> Any:
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket

        from .mongo_connection import get_connection

        conn = await get_connection(uri=settings.mongo_uri, db_name=settings.mongo_db_name)
        db = conn.get_database()
        return AsyncIOMotorGridFSBucket(db, bucket_name=self._bucket_name)

    async def save(
        self,
        filename: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredFile:
        bucket = await self._bucket()
        meta = {"content_type": content_type, **(metadata or {})}
        file_id = await bucket.upload_from_stream(filename, data, metadata=meta)
        logger.info(
            "FileStore(mongodb): stored %r (%d bytes) as GridFS id %s",
            filename, len(data), file_id,
        )
        return StoredFile(
            file_id=str(file_id),
            filename=filename,
            size=len(data),
            backend=self.backend,
            content_type=content_type,
            uploaded_by=meta.get("uploaded_by") or None,
        )

    async def read(self, file_id: str) -> bytes:
        from bson import ObjectId

        bucket = await self._bucket()
        stream = await bucket.open_download_stream(ObjectId(file_id))
        return await stream.read()

    async def delete(self, file_id: str) -> bool:
        from bson import ObjectId

        bucket = await self._bucket()
        try:
            await bucket.delete(ObjectId(file_id))
            return True
        except Exception as exc:  # noqa: BLE001 — unknown id / already gone
            logger.warning("FileStore(mongodb): delete %s failed: %s", file_id, exc)
            return False

    async def list(self, limit: int = 100) -> list[StoredFile]:
        from .mongo_connection import get_connection

        conn = await get_connection(uri=settings.mongo_uri, db_name=settings.mongo_db_name)
        files_col = conn.get_collection(f"{self._bucket_name}.files")
        cursor = files_col.find({}, sort=[("uploadDate", -1)], limit=limit)
        out: list[StoredFile] = []
        async for doc in cursor:
            meta = doc.get("metadata") or {}
            upload_date = doc.get("uploadDate")
            out.append(
                StoredFile(
                    file_id=str(doc["_id"]),
                    filename=doc.get("filename", ""),
                    size=int(doc.get("length", 0)),
                    backend=self.backend,
                    content_type=meta.get("content_type"),
                    uploaded_at=upload_date.isoformat() if upload_date else None,
                    uploaded_by=meta.get("uploaded_by") or None,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_file_store() -> FileStore:
    """Return the file store — always MongoDB GridFS (the only backend)."""
    return MongoGridFSFileStore()

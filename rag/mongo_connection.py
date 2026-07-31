"""MongoDB connection manager for the Knowledge Service.

Wraps a Motor async client with:
  - Connection pooling (sized from RAG_MONGO_MAX_POOL_SIZE)
  - Automatic index creation on first use (idempotent)
  - Ping / health check
  - Graceful shutdown

One MongoConnection per (uri, db_name) pair is cached at module level so every
caller (vector store, parent store, graph store, file store) shares the same
pool rather than opening new sockets. Ported from the LLM Gateway's
``features/mongo_connection.py`` — same mechanism, now backed by this
service's own ``RAGSettings`` instead of ``LLMGatewaySettings``.

Usage::

    from rag.mongo_connection import get_connection

    conn = await get_connection()              # uses settings defaults
    col  = conn.get_collection("chunks")
    await col.insert_one({"key": "value"})
    healthy = await conn.ping()
    # at app shutdown:
    await conn.close()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── module-level pool: one client per (uri, db) ─────────────────────────────
_pool: dict[str, MongoConnection] = {}
_pool_lock = asyncio.Lock()


async def get_connection(
    uri: str | None = None,
    db_name: str | None = None,
) -> MongoConnection:
    """Return (or create) the shared MongoConnection for this (uri, db) pair.

    Safe to call concurrently — uses asyncio.Lock to prevent duplicate clients.
    Passing uri=None / db_name=None falls back to RAGSettings values.
    """
    from .config import settings

    resolved_uri = uri or settings.mongo_uri
    resolved_db = db_name or settings.mongo_db_name
    key = f"{resolved_uri}::{resolved_db}"

    async with _pool_lock:
        if key not in _pool:
            conn = MongoConnection(uri=resolved_uri, db_name=resolved_db)
            await conn.connect()
            _pool[key] = conn
            logger.info("MongoDB connection established: db=%s", resolved_db)
        return _pool[key]


async def close_all() -> None:
    """Close every pooled connection (call from app shutdown handler)."""
    async with _pool_lock:
        for conn in _pool.values():
            await conn.close()
        _pool.clear()
    logger.info("All MongoDB connections closed.")


class MongoConnection:
    """Thin wrapper around a Motor AsyncIOMotorClient.

    Responsibilities:
      - Hold the client + db reference
      - Provide get_collection() so callers never touch the client directly
      - Manage index creation (idempotent, guarded by asyncio.Lock)
      - Expose ping() for health checks
    """

    def __init__(self, uri: str, db_name: str) -> None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:
            raise RuntimeError(
                "motor is not installed. "
                "Add 'motor>=3.3' to requirements.txt and reinstall."
            ) from exc

        from .config import settings as s

        self._uri = uri
        self._db_name = db_name
        self._client = AsyncIOMotorClient(
            uri,
            # Connection pool
            maxPoolSize=s.mongo_max_pool_size,
            minPoolSize=s.mongo_min_pool_size,
            # Timeouts (ms)
            connectTimeoutMS=s.mongo_connect_timeout_ms,
            serverSelectionTimeoutMS=s.mongo_server_selection_timeout_ms,
            socketTimeoutMS=s.mongo_socket_timeout_ms,
            # Write concern: wait for primary acknowledgement
            w=1,
            # TLS — Atlas requires it; Motor enables it automatically for srv URIs
        )
        self._db = self._client[db_name]
        # Per-collection index-creation guard
        self._index_locks: dict[str, asyncio.Lock] = {}
        self._indexes_done: set[str] = set()

    # ---------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        """Force the connection and verify it is reachable."""
        await self._client.admin.command("ping")

    async def close(self) -> None:
        """Release the Motor client and all its connections."""
        self._client.close()

    # ----------------------------------------------------------------- public

    def get_collection(self, name: str) -> Any:
        """Return the Motor collection for `name` in this database."""
        return self._db[name]

    def get_database(self) -> Any:
        """Return the Motor database handle (needed for e.g. GridFS buckets)."""
        return self._db

    @property
    def db_name(self) -> str:
        return self._db_name

    async def ensure_indexes(
        self,
        collection_name: str,
        indexes: list[dict[str, Any]],
    ) -> None:
        """Create indexes on `collection_name` if not already created.

        Each entry in `indexes` is a dict with keys:
          - "keys": list of (field, direction) tuples  e.g. [("created_at", -1)]
          - "options": dict of create_index kwargs     e.g. {"unique": True}

        Idempotent — calling multiple times is safe (MongoDB deduplicates).
        Guarded by a per-collection asyncio.Lock to avoid duplicate creation.
        """
        if collection_name in self._indexes_done:
            return

        if collection_name not in self._index_locks:
            self._index_locks[collection_name] = asyncio.Lock()

        async with self._index_locks[collection_name]:
            if collection_name in self._indexes_done:
                return  # double-checked inside the lock

            col = self.get_collection(collection_name)
            for idx in indexes:
                try:
                    await col.create_index(idx["keys"], **idx.get("options", {}))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Index creation failed on %s.%s: %s",
                        self._db_name,
                        collection_name,
                        exc,
                    )
            self._indexes_done.add(collection_name)
            logger.debug("Indexes ensured on %s.%s", self._db_name, collection_name)

    async def ping(self) -> bool:
        """Return True if the server is reachable within the timeout."""
        try:
            await self._client.admin.command("ping")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MongoDB ping failed: %s", exc)
            return False

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return f"MongoConnection(db={self._db_name!r})"

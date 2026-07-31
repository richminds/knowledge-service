"""MongoDB-backed parent document store for parent-child RAG retrieval.

Parent documents (broader context groups of child chunks) are stored in a
dedicated collection so the chunks collection stays lean.

Schema (parent_docs collection):
  {
    "_id":        ObjectId,
    "parent_id":  str  (unique, derived from source+section+strategy+index),
    "content":    str,
    "metadata":   dict
  }
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)

_PARENT_INDEXES = [
    {"keys": [("parent_id", 1)], "options": {"unique": True, "name": "parent_id_unique"}},
    {"keys": [("metadata.source", 1)], "options": {"name": "parent_source_idx"}},
    {"keys": [("metadata.content_hash", 1)], "options": {"name": "parent_hash_idx"}},
]


async def _get_collection() -> Any:
    from .mongo_connection import get_connection

    conn = await get_connection(
        uri=settings.mongo_uri,
        db_name=settings.mongo_db_name,
    )
    await conn.ensure_indexes(settings.parents_collection, _PARENT_INDEXES)
    return conn.get_collection(settings.parents_collection)


class MongoParentStore:
    """Async parent document store backed by MongoDB."""

    async def save(self, parent_documents: dict[str, Document]) -> None:
        """Upsert a batch of parent documents keyed by parent_id."""
        if not parent_documents:
            return

        from pymongo import ReplaceOne

        col = await _get_collection()
        ops = [
            ReplaceOne(
                {"parent_id": parent_id},
                {
                    "parent_id": parent_id,
                    "content": doc.page_content,
                    "metadata": _serialise_metadata(doc.metadata),
                },
                upsert=True,
            )
            for parent_id, doc in parent_documents.items()
        ]
        result = await col.bulk_write(ops, ordered=False)
        logger.debug(
            "ParentStore.save: %d parent(s) upserted.",
            result.upserted_count + result.modified_count,
        )

    async def load(self, parent_id: str) -> Document | None:
        """Load a single parent document by its stable ID."""
        col = await _get_collection()
        raw = await col.find_one({"parent_id": parent_id}, {"content": 1, "metadata": 1})
        if not raw:
            return None
        return Document(
            page_content=raw.get("content", ""),
            metadata=raw.get("metadata", {}),
        )

    async def delete(self, parent_ids: list[str]) -> int:
        """Delete parent documents by ID list.  Returns the deletion count."""
        if not parent_ids:
            return 0
        col = await _get_collection()
        result = await col.delete_many({"parent_id": {"$in": parent_ids}})
        logger.debug("ParentStore.delete: %d parent(s) removed.", result.deleted_count)
        return result.deleted_count

    async def delete_by_source(self, source: str) -> int:
        """Delete all parent documents for a given source file path."""
        col = await _get_collection()
        result = await col.delete_many({"metadata.source": source})
        logger.info(
            "ParentStore: deleted %d parent(s) for source '%s'.", result.deleted_count, source
        )
        return result.deleted_count


# ──────────────────────────────────────────── backward-compatible sync shim
# graph.py calls the async MongoParentStore directly; these thin helpers
# exist so any remaining sync code can import them without breakage.

async def save_parent_documents(parent_documents: dict[str, Document]) -> None:
    """Async wrapper for MongoParentStore.save — preferred entry point."""
    await MongoParentStore().save(parent_documents)


async def load_parent_document(parent_id: str) -> Document | None:
    """Async wrapper for MongoParentStore.load — preferred entry point."""
    return await MongoParentStore().load(parent_id)


def _serialise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
        for k, v in metadata.items()
    }

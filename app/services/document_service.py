"""Document/file management orchestration — deletion and file listing."""
from __future__ import annotations

import asyncio
import logging

from rag.embeddings import get_embeddings
from rag.file_store import get_file_store
from rag.graph_store import GraphStore
from rag.parent_store import MongoParentStore
from rag.vector_store import VectorStore

from ..models.document_model import DeleteResponse, FileEntry, FilesListResponse

logger = logging.getLogger(__name__)


async def delete_document(source: str) -> DeleteResponse:
    """Remove all vector chunks, parent docs, and graph nodes for a given source path."""
    embeddings = get_embeddings()
    vector_store = VectorStore(embeddings)
    parent_store = MongoParentStore()
    graph_store = GraphStore()

    chunks_deleted, parents_deleted, _ = await asyncio.gather(
        vector_store.delete_by_source(source),
        parent_store.delete_by_source(source),
        graph_store.delete_by_source(source),
    )

    logger.info(
        "Deleted source '%s': %d chunks, %d parents.", source, chunks_deleted, parents_deleted
    )
    return DeleteResponse(
        source=source, chunks_deleted=chunks_deleted, parents_deleted=parents_deleted
    )


async def list_files(limit: int) -> FilesListResponse:
    """Return files persisted in the configured file store (default: GridFS)."""
    store = get_file_store()
    files = await store.list(limit=limit)
    return FilesListResponse(
        backend=store.backend,
        count=len(files),
        files=[FileEntry(**f.model_dump()) for f in files],
    )

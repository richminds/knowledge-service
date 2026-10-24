import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas import IngestionJobStatusResponse, JobStatus
from app.rag.cache import AnswerCache
from app.rag.graph.base import GraphStore
from app.rag.ingestion.filetypes import safe_relative_path
from app.rag.ingestion.pipeline import IngestionPipeline, IngestionStats
from app.rag.vectorstore.base import VectorStore

logger = get_logger(__name__)


class IngestionService:
    """Coordinates ingestion runs, deletions and job status.

    Job state is kept in-memory for this template, so it is lost on restart and not
    shared between instances. For production scale, run ingestion from the CLI
    (scripts/ingest_cli.py) with the Mongo ingestion manifest (MANIFEST_STORE=mongo).
    """

    def __init__(
        self,
        pipeline: IngestionPipeline,
        vector_store: VectorStore | None,
        graph_store: GraphStore | None = None,
        cache: AnswerCache | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._cache = cache
        self._jobs: dict[str, IngestionJobStatusResponse] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = IngestionJobStatusResponse(
            job_id=job_id, status=JobStatus.PENDING, created_at=datetime.now(UTC)
        )
        return job_id

    def get_job(self, job_id: str) -> IngestionJobStatusResponse | None:
        return self._jobs.get(job_id)

    def _update(self, job_id: str, **changes) -> None:
        self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    async def _invalidate(self, collection: str) -> None:
        # Cached answers may cite content that just changed (C-165 freshness).
        if self._cache is not None:
            await self._cache.invalidate(collection)

    async def run_ingestion(
        self,
        job_id: str,
        source_uri: str,
        collection: str,
        metadata: dict[str, str],
        targets: Sequence[str] | None = None,
        acl: Sequence[str] | None = None,
        full_sync: bool = False,
        force: bool = False,
    ) -> None:
        self._update(job_id, status=JobStatus.RUNNING)
        try:
            stats = await self._pipeline.run(
                source_uri,
                collection,
                {**metadata, "job_id": job_id},
                targets,
                acl=acl,
                full_sync=full_sync,
                force=force,
            )
            await self._invalidate(collection)
            self._update(
                job_id,
                status=JobStatus.COMPLETED,
                targets=stats.targets,
                document_count=stats.document_count,
                chunk_count=stats.chunk_count,
                entity_count=stats.entity_count,
                relation_count=stats.relation_count,
                unchanged=len(stats.unchanged),
                acl_updated=len(stats.acl_updated),
                deleted=len(stats.deleted),
                duplicates=stats.duplicates,
                quarantined=stats.quarantined,
                failed_sources=stats.failed,
                reconciliation=stats.reconciliation,
                finished_at=datetime.now(UTC),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingestion_job_failed", job_id=job_id)
            self._update(
                job_id, status=JobStatus.FAILED, detail=str(exc), finished_at=datetime.now(UTC)
            )

    async def ingest_upload(
        self,
        filename: str,
        data: bytes,
        collection: str,
        metadata: dict[str, str],
        targets: Sequence[str] | None = None,
        acl: Sequence[str] | None = None,
        relative_path: str = "",
    ) -> tuple[str, IngestionStats]:
        stats = await self._pipeline.run_for_upload(
            filename, data, collection, metadata, targets, acl=acl, relative_path=relative_path
        )
        await self._invalidate(collection)
        name = Path(filename).name
        path = safe_relative_path(relative_path, name) if relative_path else name
        return f"upload://{path}", stats

    async def delete_source(self, source: str, collection: str) -> tuple[int, int]:
        """Remove one document everywhere: vector index, graph, manifest, answer cache
        (C-144, C-202)."""
        chunks, nodes = await self._pipeline.delete_source(source, collection)
        await self._invalidate(collection)
        return chunks, nodes

    async def manifest(self, collection: str, prefix: str = "", status: str | None = None) -> list:
        entries = await self._pipeline.manifest.list_entries(collection, prefix)
        return [e for e in entries if status is None or e.status == status]

    async def delete_collection(self, collection: str) -> tuple[int, int]:
        """Delete a collection from every configured store: (vector chunks, graph nodes)."""
        chunks = await self._vector_store.delete_collection(collection) if self._vector_store else 0
        nodes = await self._graph_store.delete_collection(collection) if self._graph_store else 0
        await self._invalidate(collection)
        return chunks, nodes

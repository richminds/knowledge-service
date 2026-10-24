import json
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)

from app.api.deps import IngestionServiceDep, SettingsDep
from app.core.audit import audit
from app.core.auth import IngestionUser
from app.rag.ingestion.filetypes import upload_problem
from app.models.schemas import (
    DeleteCollectionResponse,
    IngestionJobStatusResponse,
    ManifestEntryModel,
    IngestionRequest,
    IngestionResponse,
    JobStatus,
    UploadIngestionResponse,
)

router = APIRouter()

COLLECTION_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"


def _csv(value: str) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] or None


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionServiceDep,
    user: IngestionUser,
) -> IngestionResponse:
    """Async ingestion job (local path or connector URI) into the vector and/or graph store."""
    job_id = ingestion_service.create_job()
    background_tasks.add_task(
        ingestion_service.run_ingestion,
        job_id,
        request.source_uri,
        request.collection,
        request.metadata,
        request.targets,
        request.acl,
        request.full_sync,
        request.force,
    )
    audit(
        "ingest",
        user_id=user.user_id,
        job_id=job_id,
        source=request.source_uri,
        collection=request.collection,
    )
    return IngestionResponse(job_id=job_id, status=JobStatus.PENDING)


@router.post("/upload", response_model=UploadIngestionResponse)
async def upload_and_ingest(
    ingestion_service: IngestionServiceDep,
    settings: SettingsDep,
    user: IngestionUser,
    file: Annotated[
        UploadFile,
        File(description="One file of a type in INGESTION_UPLOAD_EXTENSIONS (PDF, DOCX, ...)"),
    ],
    collection: Annotated[str, Form(pattern=COLLECTION_PATTERN)] = "default",
    metadata: Annotated[str, Form(description="JSON object stored on every chunk")] = "{}",
    targets: Annotated[
        str, Form(description="Comma-separated: vector,graph. Empty = INGESTION_TARGETS")
    ] = "",
    acl: Annotated[
        str,
        Form(description="Comma-separated principal ids allowed to read it; empty = default"),
    ] = "",
    relative_path: Annotated[
        str,
        Form(
            max_length=1024,
            description="Path inside an uploaded folder (e.g. policies/hr/leave.pdf); keeps "
            "same-named files in different folders apart. Empty = the file name.",
        ),
    ] = "",
) -> UploadIngestionResponse:
    """Synchronously ingest a single uploaded file (small documents / interactive use)."""
    try:
        parsed_metadata = {str(k): str(v) for k, v in json.loads(metadata).items()}
    except (ValueError, AttributeError) as exc:
        raise HTTPException(422, "metadata must be a JSON object") from exc
    target_list = _csv(targets)
    if target_list and not set(target_list) <= {"vector", "graph"}:
        raise HTTPException(422, "targets must be vector and/or graph")

    max_bytes = settings.ingestion_max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.ingestion_max_upload_mb} MB")
    filename = file.filename or "upload.txt"
    if problem := upload_problem(filename, data, settings.ingestion_upload_extensions):
        raise HTTPException(415, problem)

    source, stats = await ingestion_service.ingest_upload(
        filename,
        data,
        collection,
        parsed_metadata,
        target_list,
        _csv(acl),
        relative_path=relative_path,
    )
    audit("ingest_upload", user_id=user.user_id, source=source, collection=collection)
    return UploadIngestionResponse(
        source=source,
        collection=collection,
        targets=stats.targets,
        chunk_count=stats.chunk_count,
        entity_count=stats.entity_count,
        relation_count=stats.relation_count,
    )


@router.get("/manifest", response_model=list[ManifestEntryModel])
async def list_manifest(
    ingestion_service: IngestionServiceDep,
    user: IngestionUser,
    collection: Annotated[str, Query(pattern=COLLECTION_PATTERN)] = "default",
    prefix: str = "",
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> list[ManifestEntryModel]:
    """Per-document ingestion status (C-148): indexed / skipped / quarantined (with reasons)
    / failed / deleted, with lineage (run id) and counts."""
    entries = await ingestion_service.manifest(collection, prefix, status_filter)
    return [ManifestEntryModel.model_validate(e, from_attributes=True) for e in entries]


@router.get("/{job_id}", response_model=IngestionJobStatusResponse)
async def get_ingestion_status(
    job_id: str, ingestion_service: IngestionServiceDep, user: IngestionUser
) -> IngestionJobStatusResponse:
    job = ingestion_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found")
    return job


@router.delete("/sources", response_model=DeleteCollectionResponse)
async def delete_source(
    ingestion_service: IngestionServiceDep,
    user: IngestionUser,
    source: Annotated[str, Query(min_length=1, max_length=2048)],
    collection: Annotated[str, Query(pattern=COLLECTION_PATTERN)] = "default",
) -> DeleteCollectionResponse:
    """Delete one document from the vector index, the graph and the answer cache
    (document removal / right to be forgotten, C-144, C-202)."""
    chunks, nodes = await ingestion_service.delete_source(source, collection)
    audit("delete_source", user_id=user.user_id, source=source, collection=collection)
    return DeleteCollectionResponse(
        collection=collection, deleted_chunks=chunks, deleted_graph_nodes=nodes
    )


@router.delete("/collections/{collection}", response_model=DeleteCollectionResponse)
async def delete_collection(
    collection: Annotated[str, Path(pattern=COLLECTION_PATTERN)],
    ingestion_service: IngestionServiceDep,
    user: IngestionUser,
) -> DeleteCollectionResponse:
    """Remove a collection from the vector and graph stores (e.g. before a full re-index)."""
    chunks, nodes = await ingestion_service.delete_collection(collection)
    audit("delete_collection", user_id=user.user_id, collection=collection)
    return DeleteCollectionResponse(
        collection=collection, deleted_chunks=chunks, deleted_graph_nodes=nodes
    )

"""HTTP client for the Knowledge Service.

Drop this into any application that wants document ingestion and RAG query
without embedding the retrieval pipeline itself — Portless (the application
this service's RAG pipeline was extracted from) or any future application.
Mirrors the calling convention of the LLM Gateway's own ``sdk/client.py``:
one credential, one connection pool, plain dataclasses back.

Usage::

    from sdk import KnowledgeServiceClient

    client = KnowledgeServiceClient(
        base_url="https://knowledge-service.internal",
        api_key="sk-live-...",       # identifies your application
    )

    job = await client.ingest(["/data/docs/policy.pdf"])
    status = await client.job_status(job.job_id)

    answer = await client.query("What is the refund policy?")
    print(answer.answer, answer.sources)

    await client.aclose()

The application never talks to MongoDB or an LLM provider directly — this
service owns both, exactly as this service itself never holds a provider
credential and instead calls an LLM Gateway over HTTP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0


@dataclass
class IngestResult:
    job_id: str
    status: str
    message: str


@dataclass
class JobStatus:
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    inserted_count: int | None = None
    graph_inserted_count: int | None = None
    error: str | None = None
    started_at: str = ""
    completed_at: str | None = None


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citation_validation: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeleteResult:
    source: str
    chunks_deleted: int
    parents_deleted: int


class KnowledgeServiceError(Exception):
    """Raised when the Knowledge Service returns a non-2xx response."""

    def __init__(self, status_code: int, code: str, message: str, request_id: str = "") -> None:
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        suffix = f" (id={request_id})" if request_id else ""
        super().__init__(f"[{status_code} {code}] {message}" + suffix)


def _raise_for_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    code, message, request_id = "http_error", response.text[:500], ""
    try:
        payload = response.json()
        err = payload.get("error") or {}
        code = err.get("code", code)
        message = err.get("message", payload.get("detail", message))
        request_id = err.get("request_id", "")
    except Exception:  # noqa: BLE001 — non-JSON body; keep the raw text
        pass
    raise KnowledgeServiceError(response.status_code, code, str(message), request_id)


class KnowledgeServiceClient:
    """One instance owns one connection pool — build it once per process and reuse it."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        jwt: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if jwt:
            headers["Authorization"] = f"Bearer {jwt}"

        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=timeout
        )
        self._owns_client = client is None

    # ------------------------------------------------------------ ingestion

    async def ingest(
        self,
        input_paths: list[str],
        chunk_strategy: str = "recursive",
        user_id: str | None = None,
    ) -> IngestResult:
        """Start ingestion of server-side file/directory paths.

        ``user_id`` is the end-user this ingestion is on behalf of — recorded
        as ``uploaded_by`` on every resulting chunk's metadata. Leave unset to
        record this client's own principal instead (resolved server-side).
        """
        response = await self._client.post(
            "/v1/ingest",
            json={
                "input_paths": input_paths,
                "chunk_strategy": chunk_strategy,
                "user_id": user_id,
            },
        )
        _raise_for_error(response)
        return IngestResult(**response.json())

    async def upload(
        self,
        files: list[tuple[str, bytes, str | None]],
        chunk_strategy: str = "recursive",
        user_id: str | None = None,
    ) -> IngestResult:
        """Upload files (filename, bytes, content_type) and ingest them.

        See ``ingest()`` for what ``user_id`` records.
        """
        multipart = [
            ("files", (name, data, content_type or "application/octet-stream"))
            for name, data, content_type in files
        ]
        form_data: dict[str, str] = {"chunk_strategy": chunk_strategy}
        if user_id:
            form_data["user_id"] = user_id
        response = await self._client.post(
            "/v1/upload",
            files=multipart,
            data=form_data,
        )
        _raise_for_error(response)
        return IngestResult(**response.json())

    async def job_status(self, job_id: str) -> JobStatus:
        response = await self._client.get(f"/v1/ingest/{job_id}")
        _raise_for_error(response)
        return JobStatus(**response.json())

    # ---------------------------------------------------------------- query

    async def query(
        self,
        question: str,
        metadata_filter: dict[str, Any] | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
    ) -> QueryResult:
        response = await self._client.post(
            "/v1/query",
            json={
                "question": question,
                "metadata_filter": metadata_filter,
                "user_id": user_id,
                "team_id": team_id,
            },
        )
        _raise_for_error(response)
        return QueryResult(**response.json())

    # ----------------------------------------------------------- documents

    async def delete_document(self, source: str) -> DeleteResult:
        response = await self._client.delete(f"/v1/documents/{source}")
        _raise_for_error(response)
        return DeleteResult(**response.json())

    async def list_files(self, limit: int = 100) -> dict[str, Any]:
        response = await self._client.get("/v1/files", params={"limit": limit})
        _raise_for_error(response)
        return response.json()

    # ------------------------------------------------------------- telemetry

    async def stats(self) -> dict[str, Any]:
        response = await self._client.get("/v1/stats")
        _raise_for_error(response)
        return response.json()

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/health")
        return response.json()

    # ------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> KnowledgeServiceClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

"""Integration test for the ingestion endpoints.

    POST /v1/ingest             POST /v1/upload             GET /v1/ingest/{job_id}

Only ``rag.ingestion.ingest()`` (the actual pipeline) is mocked — routing,
auth, the in-memory job registry, and the user_id/principal fallback
(app/dependencies.py::resolve_user_id) all run for real. FastAPI's
BackgroundTasks execute synchronously within the ASGI call TestClient awaits,
so by the time a `post()` call returns here, the background ingestion job has
already run to completion — no polling loop needed.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def mock_ingest(monkeypatch):
    calls: list[dict] = []

    async def _fake(input_paths, chunk_strategy="recursive", uploaded_by=None):
        calls.append(
            {
                "input_paths": input_paths,
                "chunk_strategy": chunk_strategy,
                "uploaded_by": uploaded_by,
            }
        )
        return {"inserted_count": 3, "graph_inserted_count": 0}

    monkeypatch.setattr("app.services.ingest_service.ingest", _fake)
    return calls


class _FakeFileStore:
    """Stand-in for MongoGridFSFileStore — no real Mongo connection needed."""

    backend = "fake"

    async def save(self, filename, data, content_type=None, metadata=None):
        from rag.file_store import StoredFile

        return StoredFile(
            file_id="fake-id", filename=filename, size=len(data), backend=self.backend
        )


@pytest.fixture()
def mock_file_store(monkeypatch):
    monkeypatch.setattr("app.services.ingest_service.get_file_store", lambda: _FakeFileStore())


def test_ingest_records_explicit_user_id(api, mock_ingest):
    resp = api.post(
        "/v1/ingest", json={"input_paths": ["/data/doc.md"], "user_id": "alice"}
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    assert mock_ingest[0]["uploaded_by"] == "alice"

    status_resp = api.get(f"/v1/ingest/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "completed"
    assert status_resp.json()["inserted_count"] == 3


def test_ingest_falls_back_to_principal_when_no_user_id(api, mock_ingest):
    resp = api.post("/v1/ingest", json={"input_paths": ["/data/doc.md"]})
    assert resp.status_code == 202
    # Open-mode test environment (see conftest.py) -> principal is "anonymous".
    assert mock_ingest[0]["uploaded_by"] == "anonymous"


def test_upload_records_explicit_user_id(api, mock_ingest, mock_file_store):
    resp = api.post(
        "/v1/upload",
        files={"files": ("policy.md", b"# Policy\n\nRefunds take five days.", "text/markdown")},
        data={"chunk_strategy": "recursive", "user_id": "bob"},
    )
    assert resp.status_code == 202
    assert mock_ingest[0]["uploaded_by"] == "bob"


def test_upload_falls_back_to_principal_when_no_user_id(api, mock_ingest, mock_file_store):
    resp = api.post(
        "/v1/upload",
        files={"files": ("policy.md", b"# Policy\n\nRefunds take five days.", "text/markdown")},
    )
    assert resp.status_code == 202
    assert mock_ingest[0]["uploaded_by"] == "anonymous"


def test_job_status_404_for_unknown_job(api):
    resp = api.get("/v1/ingest/does-not-exist")
    assert resp.status_code == 404

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
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_ingest(monkeypatch):
    calls: list[dict] = []

    async def _fake(
        input_paths,
        chunk_strategy="recursive",
        uploaded_by=None,
        org_id=None,
        account_id=None,
    ):
        calls.append(
            {
                "input_paths": input_paths,
                "chunk_strategy": chunk_strategy,
                "uploaded_by": uploaded_by,
                "org_id": org_id,
                "account_id": account_id,
            }
        )
        return {"inserted_count": 3, "graph_inserted_count": 0}

    monkeypatch.setattr("app.services.ingest_service.ingest", _fake)
    return calls


class _FakeFileStore:
    """Stand-in for MongoGridFSFileStore — no real Mongo connection needed."""

    backend = "fake"

    def __init__(self) -> None:
        self.save_calls: list[dict] = []

    async def save(self, filename, data, content_type=None, metadata=None):
        from rag.file_store import StoredFile

        self.save_calls.append({"filename": filename, "metadata": metadata or {}})
        return StoredFile(
            file_id="fake-id",
            filename=filename,
            size=len(data),
            backend=self.backend,
            org_id=(metadata or {}).get("org_id") or None,
            account_id=(metadata or {}).get("account_id") or None,
        )


@pytest.fixture()
def mock_file_store(monkeypatch):
    store = _FakeFileStore()
    monkeypatch.setattr("app.services.ingest_service.get_file_store", lambda: store)
    return store


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


def test_ingest_records_explicit_org_id(api, mock_ingest):
    resp = api.post(
        "/v1/ingest", json={"input_paths": ["/data/doc.md"], "org_id": "org-a"}
    )
    assert resp.status_code == 202
    assert mock_ingest[0]["org_id"] == "org-a"


def test_ingest_defaults_org_id_to_empty_string_when_omitted(api, mock_ingest):
    resp = api.post("/v1/ingest", json={"input_paths": ["/data/doc.md"]})
    assert resp.status_code == 202
    assert mock_ingest[0]["org_id"] == ""


def test_upload_records_explicit_org_id(api, mock_ingest, mock_file_store):
    resp = api.post(
        "/v1/upload",
        files={"files": ("policy.md", b"# Policy\n\nRefunds take five days.", "text/markdown")},
        data={"chunk_strategy": "recursive", "org_id": "org-a"},
    )
    assert resp.status_code == 202
    assert mock_ingest[0]["org_id"] == "org-a"
    # The persisted file's own metadata (read back by GET /v1/files) carries
    # org_id too, not just the ingestion pipeline call.
    assert mock_file_store.save_calls[0]["metadata"]["org_id"] == "org-a"


# ── Gateway-authenticated caller: identity is the gateway's verified header,
# not the request body (app/dependencies.py::resolve_user_id / resolve_account_id)
# — this is the tenant-isolation fix.


def _gateway_client(monkeypatch) -> TestClient:
    import rag.config as rag_config_mod

    monkeypatch.setattr(rag_config_mod.settings, "auth_enabled", True)

    from app.main import create_app

    return TestClient(create_app())


def _gateway_headers(user_id: str = "USR-1", **extra: str) -> dict[str, str]:
    """What the API gateway injects once it has verified the caller."""
    headers = {"X-User-ID": user_id, "X-Authenticated-Via": "api-gateway"}
    if extra.get("account_id"):
        headers["X-Account-ID"] = extra["account_id"]
    return headers


def test_ingest_gateway_derives_identity_from_headers(monkeypatch, mock_ingest):
    with _gateway_client(monkeypatch) as client:
        resp = client.post(
            "/v1/ingest",
            json={"input_paths": ["/data/doc.md"]},
            headers=_gateway_headers(account_id="org-a"),
        )
        assert resp.status_code == 202
        assert mock_ingest[0]["org_id"] == "org-a"
        assert mock_ingest[0]["uploaded_by"] == "USR-1"


def test_ingest_gateway_rejects_org_id_that_does_not_match(monkeypatch, mock_ingest):
    with _gateway_client(monkeypatch) as client:
        resp = client.post(
            "/v1/ingest",
            json={"input_paths": ["/data/doc.md"], "org_id": "org-b"},
            headers=_gateway_headers(account_id="org-a"),
        )
        assert resp.status_code == 403
        assert mock_ingest == []  # never reached the pipeline


def test_ingest_gateway_rejects_user_id_that_does_not_match(monkeypatch, mock_ingest):
    with _gateway_client(monkeypatch) as client:
        resp = client.post(
            "/v1/ingest",
            json={"input_paths": ["/data/doc.md"], "user_id": "someone-else"},
            headers=_gateway_headers(account_id="org-a"),
        )
        assert resp.status_code == 403
        assert mock_ingest == []

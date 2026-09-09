"""Integration test for GET /v1/files — persisted-file listing + org_id scoping.

Only the FileStore (GridFS) is mocked; routing, auth, and
app/services/document_service.py all run for real.
"""
from __future__ import annotations

import pytest

from rag.file_store import StoredFile


class _FakeFileStore:
    backend = "fake"

    def __init__(self, files: list[StoredFile]) -> None:
        self._files = files

    async def list(
        self,
        limit: int = 100,
        org_id: str | None = None,
        account_id: str | None = None,
    ) -> list[StoredFile]:
        # Mirrors MongoGridFSFileStore.list()'s filtering rule so this test
        # exercises the same contract the real backend promises.
        def visible(f: StoredFile) -> bool:
            if f.org_id not in (None, "*") and f.org_id != org_id:
                return False
            return f.account_id in (None, "*") or f.account_id == account_id

        return [f for f in self._files if visible(f)][:limit]


@pytest.fixture()
def files_fixture():
    return [
        StoredFile(file_id="1", filename="org-a.md", size=10, backend="fake", org_id="org-a"),
        StoredFile(file_id="2", filename="org-b.md", size=10, backend="fake", org_id="org-b"),
        StoredFile(file_id="3", filename="public.md", size=10, backend="fake", org_id=None),
    ]


def test_list_files_scoped_to_caller_org(api, monkeypatch, files_fixture):
    monkeypatch.setattr(
        "app.services.document_service.get_file_store", lambda: _FakeFileStore(files_fixture)
    )

    resp = api.get("/v1/files", params={"org_id": "org-a"})
    assert resp.status_code == 200
    filenames = {f["filename"] for f in resp.json()["files"]}
    assert filenames == {"org-a.md", "public.md"}


def test_list_files_without_org_id_shows_only_unscoped(api, monkeypatch, files_fixture):
    monkeypatch.setattr(
        "app.services.document_service.get_file_store", lambda: _FakeFileStore(files_fixture)
    )

    resp = api.get("/v1/files")
    assert resp.status_code == 200
    filenames = {f["filename"] for f in resp.json()["files"]}
    assert filenames == {"public.md"}

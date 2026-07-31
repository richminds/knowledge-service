"""Request/response schemas for document and file management endpoints."""
from __future__ import annotations

from pydantic import BaseModel


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int
    parents_deleted: int


class FileEntry(BaseModel):
    file_id: str
    filename: str
    size: int
    backend: str
    content_type: str | None = None
    uploaded_at: str | None = None


class FilesListResponse(BaseModel):
    backend: str
    count: int
    files: list[FileEntry]

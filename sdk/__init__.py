"""HTTP SDK for the Knowledge Service — see sdk/client.py for usage."""
from .client import (
    DeleteResult,
    IngestResult,
    JobStatus,
    KnowledgeServiceClient,
    KnowledgeServiceError,
    QueryResult,
)

__all__ = [
    "KnowledgeServiceClient",
    "KnowledgeServiceError",
    "IngestResult",
    "JobStatus",
    "QueryResult",
    "DeleteResult",
]

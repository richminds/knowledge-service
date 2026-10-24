from fastapi import Request, status
from fastapi.responses import JSONResponse


class RagAgentError(Exception):
    """Base class for all application-raised errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "Internal server error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message


class DocumentLoadError(RagAgentError):
    status_code = 422  # Unprocessable Content
    message = "Failed to load or parse the source document"


class IngestionConfigError(RagAgentError):
    status_code = status.HTTP_400_BAD_REQUEST
    message = "Ingestion target is not configured"


class EmbeddingError(RagAgentError):
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "Failed to generate embeddings"


class VectorStoreError(RagAgentError):
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "Vector store operation failed"


class VectorBackendError(RagAgentError):
    status_code = status.HTTP_400_BAD_REQUEST
    message = "Requested vector store is not available"


class GraphStoreError(RagAgentError):
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "Graph store operation failed"


class RagModeError(RagAgentError):
    status_code = status.HTTP_400_BAD_REQUEST
    message = "Requested RAG mode is not available"


class GenerationError(RagAgentError):
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "Failed to generate an answer from the language model"


async def rag_agent_exception_handler(request: Request, exc: RagAgentError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.message},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the traceback, never leak internals to the client."""
    from app.core.logging import get_logger

    get_logger(__name__).exception("unhandled_exception", path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "detail": "An unexpected error occurred"},
    )

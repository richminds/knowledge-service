"""Domain exception → HTTP status mapping.

The core (``rag/``) stays framework-free and raises plain exceptions or lets
FastAPI's own ``HTTPException`` do the talking at the service boundary; this
is the only place that decides what each domain exception means over HTTP.

    GatewayError (rag/llm_gateway_sdk.py) → forwards the LLM Gateway's own
        status code (503 all models failed, 429 caller throttled, 500
        gateway misconfigured, ...) instead of collapsing every downstream
        failure into an opaque 500 — the whole point of surfacing this
        service's own dependency status accurately.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.llm_gateway_sdk import GatewayError

logger = logging.getLogger(__name__)


def _problem(
    request: Request,
    status: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", ""),
            **extra,
        }
    }
    return JSONResponse(status_code=status, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        logger.warning(
            "LLM Gateway call failed: status=%s code=%s message=%s",
            exc.status_code, exc.code, exc,
        )
        headers = {"Retry-After": "5"} if exc.status_code in (429, 503) else None
        return _problem(
            request,
            exc.status_code,
            f"llm_gateway_{exc.code}",
            str(exc),
            headers=headers,
            upstream_request_id=exc.request_id,
        )

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger

logger = get_logger("app.access")

REQUEST_ID_HEADER = "x-request-id"


def current_request_id() -> str:
    """The X-Request-ID of the request being served (a fresh id outside a request)."""
    return structlog.contextvars.get_contextvars().get("request_id") or uuid.uuid4().hex


class RequestContextMiddleware:
    """Assigns/propagates X-Request-ID, binds it to every log line, and logs request timing.

    Pure ASGI (not BaseHTTPMiddleware) so it doesn't buffer streaming (SSE) responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = headers.get(REQUEST_ID_HEADER.encode(), b"").decode() or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER.encode(), request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "http_request",
                method=scope["method"],
                path=scope["path"],
                status=status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )

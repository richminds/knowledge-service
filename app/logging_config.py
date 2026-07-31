"""Structured (JSON) logging setup — the log-aggregator-friendly default.

One JSON object per log line: every log aggregator built in the last decade
(Datadog, Loki, CloudWatch Logs Insights, ELK/OpenSearch, Google Cloud
Logging) parses JSON-per-line natively and indexes its fields, where a plain
text line needs a hand-written regex per deployment to get the same query
power. This needs no new external service — stdout stays the transport
(``PYTHONUNBUFFERED=1`` is set in the Dockerfile), only the encoding of each
line changes. Ported from the LLM Gateway's ``app/logging_config.py``.

Every record automatically carries the ambient correlation ID from
``rag/log_context.py`` (when one is bound) as ``request_id``, so a single
grep/query across log lines finds everything for one HTTP call.

``KNOWLEDGE_LOG_FORMAT=text`` reverts to a plain-text formatter (e.g. for a
local dev terminal where a human is reading it live) — JSON is the default
because it is the right choice for anything that ships logs somewhere.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from rag.log_context import get_request_id

# Attributes every stdlib LogRecord already has — anything else passed via
# logger.info(..., extra={...}) is assumed to be a deliberately-added
# structured field and gets folded into the JSON object.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JSONFormatter(logging.Formatter):
    """One JSON object per log line.

    Fixed fields: ``timestamp`` (ISO 8601 UTC), ``level``, ``logger``,
    ``message``, ``request_id`` (from the ambient contextvar, "" when unset).
    Anything passed via ``extra={...}`` on the logging call is merged in
    as-is, so call sites can attach request-specific fields
    (``method``, ``path``, ``status``, ``latency_ms``, ...) without this
    formatter needing to know about them in advance.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # ISO 8601 with milliseconds, UTC — the one timestamp shape every log
        # aggregator's auto-detection recognises without a custom pattern.
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + (
            f".{int(record.msecs):03d}Z"
        )


_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str, log_format: str = "json") -> None:
    """Install this module's log handler on the root logger. Safe to call
    more than once (e.g. every app startup in a test suite that builds the
    app repeatedly) and safe to call alongside other tooling that also
    attaches its own root handler — such as pytest's ``caplog`` fixture.

    ``log_format``: "json" (default — see module docstring for why) or
    "text" (for a human reading a local terminal live).

    Only removes/replaces a handler THIS function installed previously
    (marked via an attribute on the handler instance) — it never touches
    handlers belonging to something else.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        JSONFormatter() if log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    )
    handler._knowledge_service_managed = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers = [
        h for h in root.handlers if not getattr(h, "_knowledge_service_managed", False)
    ]
    root.addHandler(handler)

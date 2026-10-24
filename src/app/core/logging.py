import logging
import sys

import structlog

# Chatty SDK loggers (every HTTP request/response at INFO) drown out application events.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "urllib3")


def configure_logging(log_level: str = "INFO", mask_pii: bool = True) -> None:
    """Configure structlog + stdlib logging to emit one JSON object per line on stdout."""
    from app.core.pii import mask_pii_processor

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # request_id, rag_mode, ...
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,  # logger.exception(...) -> "exception" field
    ]
    if mask_pii:
        # C-196: keep emails, card numbers, phone numbers... out of logs and App Insights.
        shared_processors.append(mask_pii_processor)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Plain stdlib records (uvicorn, httpx) get the same fields as structlog ones.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

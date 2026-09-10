"""Service-level configuration for the Knowledge Service HTTP API.

Env-var prefix: ``KNOWLEDGE_``.

This is deliberately separate from ``rag/config.py`` (prefix ``RAG_``): that
file owns *what the RAG pipeline does and which LLM Gateway it calls*, this
one owns *how this service is exposed*. Keeping them apart means the ``rag/``
package stays portable and the two can be copied into another service
independently — exactly the split used by the LLM Gateway this service calls
(``app/config.py`` / ``features/config.py`` there).
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


class KnowledgeServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNOWLEDGE_",
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------- server
    host: str = "0.0.0.0"
    port: int = 8090
    root_path: str = ""          # set when behind a path-prefixing proxy
    log_level: str = "INFO"
    # "json" (default) — one JSON object per log line. "text" reverts to a
    # plain format for a human reading a local terminal live. See
    # app/logging_config.py.
    log_format: str = "json"
    environment: str = "development"  # development | staging | production

    # ------------------------------------------------------------- docs
    docs_enabled: bool = True    # set false in production if the API is public

    # ------------------------------------------------------------- CORS
    # Comma-separated origins, or "*" for any. Browsers only — service-to-service
    # callers are unaffected.
    cors_origins: str = ""

    # Vercel gives every deployment and every preview its own hostname
    # (knowledge-ingest-ui-<hash>-<scope>.vercel.app), so an exact allowlist
    # goes stale on each deploy and previews are broken by default. A regex
    # covers the whole family in one rule — anchor it to the projects you
    # actually own, never leave it open.
    cors_origin_regex: str = ""

    # ------------------------------------------------------------- limits
    # Cheap structural guardrails applied before any pipeline work starts.
    max_question_length: int = 4_000
    max_upload_files_per_request: int = 20
    max_upload_file_size_mb: int = 50
    max_metadata_filter_keys: int = 20

    # ------------------------------------------------------------- misc
    request_id_header: str = "X-Request-ID"
    # Expose resolved (non-secret) configuration on GET /v1/config.
    expose_config_endpoint: bool = True

    # ---------------------------------------------------------- derived

    def parsed_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    @property
    def max_upload_file_size_bytes(self) -> int:
        return self.max_upload_file_size_mb * 1024 * 1024


gateway_settings = KnowledgeServiceSettings()

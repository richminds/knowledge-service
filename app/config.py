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

    # ------------------------------------------------------------- auth
    # Static service API keys — the simplest cloud auth, same shape as the LLM
    # Gateway's GATEWAY_API_KEYS. Comma-separated list of "name:key" or
    # "name:key:role" triples (the name is used as the default caller/principal),
    # or bare keys. role defaults to "user" when omitted — only "admin" unlocks
    # the admin routes (GET /v1/stats, /v1/config).
    #   KNOWLEDGE_API_KEYS=portless-backend:sk-live-abc,other-app:sk-live-def:admin
    # Callers send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    #
    # JWT auth is configured separately on the core settings
    # (RAG_AUTH_ENABLED / RAG_JWT_SECRET) and is checked when no API key matches.
    #
    # With NEITHER configured the API is open — fine for localhost, never for
    # a deployed service. main.py logs a loud warning at startup in that case.
    api_keys: str = ""

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

    def parsed_api_keys(self) -> dict[str, tuple[str, str]]:
        """Return {api_key: (principal_name, role)}. Empty when unconfigured.

        Accepts "key", "name:key", or "name:key:role" — role defaults to
        "user" when omitted.
        """
        out: dict[str, tuple[str, str]] = {}
        for i, raw in enumerate(p.strip() for p in self.api_keys.split(",")):
            if not raw:
                continue
            parts = [p.strip() for p in raw.split(":")]
            if len(parts) == 1:
                name, key, role = f"service-{i + 1}", parts[0], "user"
            elif len(parts) == 2:
                name, key, role = parts[0], parts[1], "user"
            else:
                name, key, role = parts[0], parts[1], (parts[2] or "user")
            if key:
                out[key] = (name or f"service-{i + 1}", role)
        return out

    def parsed_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    @property
    def max_upload_file_size_bytes(self) -> int:
        return self.max_upload_file_size_mb * 1024 * 1024


gateway_settings = KnowledgeServiceSettings()

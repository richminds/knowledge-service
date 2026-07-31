"""Local dev entrypoint — `python run.py`.

Reads host/port/log-level from KNOWLEDGE_* env vars so it matches what the
container will do. For production use the uvicorn command in the Dockerfile
instead; this exists for the reload loop.
"""
from __future__ import annotations

import uvicorn

from app.config import gateway_settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=gateway_settings.host,
        port=gateway_settings.port,
        log_level=gateway_settings.log_level.lower(),
        reload=not gateway_settings.is_production,
    )

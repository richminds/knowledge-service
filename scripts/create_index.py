"""Create the MongoDB collections, indexes and Atlas Search indexes if they do not exist.

Run from the pipeline (or by hand) when the app identity is not allowed to manage
indexes, i.e. MONGODB_CREATE_INDEXES_ON_STARTUP=false in QA/Prod.

Usage:
    python scripts/create_index.py
"""

import asyncio
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.core.config import get_settings  # noqa: E402
from app.core.mongo import close_mongo_clients  # noqa: E402
from app.rag.graph.factory import get_graph_store  # noqa: E402
from app.rag.vectorstore.factory import get_vector_store  # noqa: E402


async def main() -> None:
    settings = get_settings()
    try:
        await get_vector_store(settings).ensure_index()
        print(f"Vector collection '{settings.mongodb_chunks_collection}' is ready")
        if settings.graph_enabled:
            await get_graph_store(settings).ensure_schema()
            print(f"Graph collections '{settings.mongodb_graph_prefix}_*' are ready")
    finally:
        await close_mongo_clients()


if __name__ == "__main__":
    asyncio.run(main())

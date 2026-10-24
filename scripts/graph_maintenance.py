"""Graph maintenance jobs (C4, C5) — schedule after large ingests or nightly (G-34).

  resolve      merge near-duplicate entities (G-17, G-33)
  embed        (re)embed entity descriptions for similarity linking (G-25)
  communities  Louvain communities + PageRank + LLM reports indexed for global search
               (G-27..G-30)
  quality      counts, orphans, duplicate candidates, sample triples to review (G-24)
  all          resolve, embed, communities, quality

Backups (G-35): MongoDB Atlas takes the backups (cloud backup / point-in-time restore,
configured on the Atlas cluster); no job needed here.

Usage:
    python scripts/graph_maintenance.py all --collection default
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import get_chat_client, get_embedding_client  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.rag.graph.communities import GraphMaintenance  # noqa: E402
from app.rag.graph.factory import get_graph_store  # noqa: E402
from app.rag.vectorstore.factory import get_vector_store  # noqa: E402


async def main(task: str, collection: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = get_graph_store(settings)
    vectors = get_vector_store(settings) if settings.vector_enabled else None
    maintenance = GraphMaintenance(
        store,
        llm=get_chat_client(settings),
        embedder=get_embedding_client(settings),
        vector_store=vectors,
        acl_scope=frozenset(settings.graph_community_acl_scope),
        max_levels=settings.graph_community_levels,
        min_community_size=settings.graph_community_min_size,
    )
    try:
        if task in ("resolve", "all"):
            print("merged entities:", await maintenance.resolve_duplicates(collection))
        if task in ("embed", "all"):
            print("embedded entities:", await maintenance.embed_entities(collection))
        if task in ("communities", "all"):
            print("communities:", await maintenance.build_communities(collection))
        if task in ("quality", "all"):
            print(json.dumps(await store.quality_report(collection), indent=2, default=str))
    finally:
        await store.close()
        if vectors is not None:
            await vectors.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph maintenance")
    parser.add_argument("task", choices=["resolve", "embed", "communities", "quality", "all"])
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()
    asyncio.run(main(args.task, args.collection))

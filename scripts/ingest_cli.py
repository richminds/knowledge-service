"""CLI for running the ingestion pipeline directly, without the API.

Usage:
    python scripts/ingest_cli.py --source sample/ --collection default
    python scripts/ingest_cli.py --source blob://documents/hr/ --collection hr \
        --targets vector,graph
"""

import argparse
import asyncio
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import build_ingestion_pipeline  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.rag.graph.factory import get_graph_store  # noqa: E402
from app.rag.vectorstore.factory import get_vector_store  # noqa: E402


async def main(
    source: str, collection: str, targets: list[str] | None, full_sync: bool, force: bool
) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    resolved = targets or settings.ingestion_targets
    stores = []
    if "vector" in resolved:
        stores.append(get_vector_store(settings))
        await stores[-1].ensure_index()
    if "graph" in resolved:
        stores.append(get_graph_store(settings))
        await stores[-1].ensure_schema()

    pipeline = build_ingestion_pipeline(settings)
    try:
        stats = await pipeline.run(
            source, collection, {"ingested_by": "cli"}, resolved, full_sync=full_sync, force=force
        )
    finally:
        for store in stores:
            await store.close()

    print(
        f"Ingested {stats.document_count} documents into {stats.targets}: "
        f"{stats.chunk_count} chunks, {stats.entity_count} entities, "
        f"{stats.relation_count} relations (collection={collection})"
    )
    print(
        f"  unchanged={len(stats.unchanged)} acl_updated={len(stats.acl_updated)} "
        f"duplicates={len(stats.duplicates)} deleted={len(stats.deleted)} "
        f"reconciliation={stats.reconciliation}"
    )
    for source_name, reason in stats.quarantined.items():
        print(f"  QUARANTINED {source_name}: {reason}", file=sys.stderr)
    for failed_source, error in stats.failed.items():
        print(f"  FAILED {failed_source}: {error}", file=sys.stderr)
    return 1 if stats.failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline locally")
    parser.add_argument(
        "--source",
        required=True,
        help="Path under INGESTION_LOCAL_ROOT, or blob://<container>/<prefix>",
    )
    parser.add_argument("--collection", default="default", help="Target collection/namespace")
    parser.add_argument("--targets", default="", help="vector,graph (default: INGESTION_TARGETS)")
    parser.add_argument("--full-sync", action="store_true", help="Delete docs gone from the source")
    parser.add_argument("--force", action="store_true", help="Re-process unchanged documents")
    args = parser.parse_args()
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] or None

    sys.exit(
        asyncio.run(main(args.source, args.collection, target_list, args.full_sync, args.force))
    )

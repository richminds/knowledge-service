"""Scheduled full synchronisation (C-146): re-walks every configured source, ingests new or
changed documents, applies ACL changes and deletes documents that disappeared (C-144),
then reports count reconciliation (C-139). Schedule it nightly (Container Apps Job cron
or an Azure Pipelines schedule).

Sources are listed in config/sync_schedule.yaml (placeholder).

Usage:
    python scripts/reconcile.py [--config config/sync_schedule.yaml]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import build_ingestion_pipeline  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("reconcile")


async def main(config_path: str) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    schedule = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    pipeline = build_ingestion_pipeline(settings)
    failures = 0
    for source in schedule.get("sources") or []:
        stats = await pipeline.run(
            source["uri"],
            source.get("collection", settings.ingestion_default_collection),
            targets=source.get("targets"),
            acl=source.get("acl"),
            full_sync=True,
        )
        failures += len(stats.failed) + stats.reconciliation.get("mismatches", 0)
        print(
            f"{source['uri']}: {stats.document_count} indexed, {len(stats.unchanged)} unchanged, "
            f"{len(stats.acl_updated)} ACL updates, {len(stats.deleted)} deleted, "
            f"{len(stats.quarantined)} quarantined, {len(stats.failed)} failed, "
            f"reconciliation {stats.reconciliation}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full sync of configured sources")
    parser.add_argument("--config", default="config/sync_schedule.yaml")
    sys.exit(asyncio.run(main(parser.parse_args().config)))

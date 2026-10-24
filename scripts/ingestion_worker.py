"""Queue-driven ingestion worker (C-032..C-035, C-147).

Consumes ingestion jobs / Event Grid blob events from an Azure Service Bus queue and runs
the ingestion pipeline. Run several replicas (Container Apps Job / KEDA scaler on queue
length) to scale out. Failed messages are abandoned and retried; after the queue's
max delivery count Service Bus moves them to the dead-letter queue (C-033).

Placeholders: SERVICEBUS_NAMESPACE (<name>.servicebus.windows.net, managed identity with
"Azure Service Bus Data Receiver") or SERVICEBUS_CONNECTION_STRING; INGESTION_QUEUE_NAME.

Usage:
    python scripts/ingestion_worker.py            # run until stopped
    python scripts/ingestion_worker.py --once     # drain the queue once and exit (cron)
"""

import argparse
import asyncio
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import build_ingestion_pipeline  # noqa: E402
from app.core.azure_auth import get_azure_credential  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.rag.ingestion.jobs import parse_message  # noqa: E402

logger = get_logger("ingestion_worker")


async def handle(pipeline, body: str, default_collection: str) -> None:
    for job in parse_message(body, default_collection):
        if job.delete:
            await pipeline.delete_source(job.source_uri, job.collection)
            logger.info("job_deleted", source=job.source_uri)
            continue
        stats = await pipeline.run(
            job.source_uri,
            job.collection,
            job.metadata,
            job.targets,
            job.acl,
            full_sync=job.full_sync,
            force=job.force,
        )
        logger.info(
            "job_done",
            source=job.source_uri,
            documents=stats.document_count,
            chunks=stats.chunk_count,
            failed=len(stats.failed),
            quarantined=len(stats.quarantined),
        )


async def main(once: bool) -> None:
    from azure.servicebus import ServiceBusReceivedMessage
    from azure.servicebus.aio import ServiceBusClient

    settings = get_settings()
    configure_logging(settings.log_level)
    pipeline = build_ingestion_pipeline(settings)
    connection = settings.servicebus_connection_string.get_secret_value()
    client = (
        ServiceBusClient.from_connection_string(connection)
        if connection
        else ServiceBusClient(settings.servicebus_namespace, credential=get_azure_credential())
    )
    async with (
        client,
        client.get_queue_receiver(
            settings.ingestion_queue_name, max_wait_time=30 if once else None
        ) as receiver,
    ):
        async for message in receiver:
            msg: ServiceBusReceivedMessage = message
            try:
                await handle(pipeline, str(msg), settings.ingestion_default_collection)
                await receiver.complete_message(msg)
            except Exception:  # noqa: BLE001 - retried, then dead-lettered by Service Bus
                logger.exception(
                    "job_failed", message_id=msg.message_id, delivery_count=msg.delivery_count
                )
                await receiver.abandon_message(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Service Bus ingestion worker")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit")
    asyncio.run(main(parser.parse_args().once))

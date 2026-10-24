"""Ingestion job messages for the queue worker (C-032..C-035, C-146).

A queue message is either
  - an ingestion job: {"source_uri": ..., "collection": ..., "targets": [...], "acl": [...],
    "full_sync": bool, "force": bool, "delete": bool}
  - an Event Grid storage event (single or batch), from an Event Grid subscription on the
    documents storage account routed into the queue (C-034):
    Microsoft.Storage.BlobCreated -> ingest that blob; BlobDeleted -> delete it everywhere.
"""

import json
import re
from dataclasses import dataclass, field

_BLOB_SUBJECT = re.compile(
    r"/blobServices/default/containers/(?P<container>[^/]+)/blobs/(?P<path>.+)$"
)


@dataclass
class IngestionJob:
    source_uri: str
    collection: str = "default"
    targets: list[str] | None = None
    acl: list[str] | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    full_sync: bool = False
    force: bool = False
    delete: bool = False


def parse_message(body: str | bytes, default_collection: str = "default") -> list[IngestionJob]:
    data = json.loads(body)
    events = data if isinstance(data, list) else [data]
    jobs: list[IngestionJob] = []
    for event in events:
        if "eventType" in event and "subject" in event:
            match = _BLOB_SUBJECT.search(event["subject"])
            if not match:
                continue
            uri = f"blob://{match['container']}/{match['path']}"
            if event["eventType"] == "Microsoft.Storage.BlobCreated":
                jobs.append(IngestionJob(uri, default_collection))
            elif event["eventType"] == "Microsoft.Storage.BlobDeleted":
                jobs.append(IngestionJob(uri, default_collection, delete=True))
            continue
        jobs.append(
            IngestionJob(
                source_uri=event["source_uri"],
                collection=event.get("collection", default_collection),
                targets=event.get("targets"),
                acl=event.get("acl"),
                metadata={str(k): str(v) for k, v in (event.get("metadata") or {}).items()},
                full_sync=bool(event.get("full_sync")),
                force=bool(event.get("force")),
                delete=bool(event.get("delete")),
            )
        )
    return jobs

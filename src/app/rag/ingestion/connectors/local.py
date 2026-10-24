"""Local folder connector (C-058). Paths are confined to INGESTION_LOCAL_ROOT.

ACLs: an optional sidecar `<file>.acl.json` (a JSON list of principal ids) sets a file's
permissions; otherwise the request / INGESTION_DEFAULT_ACL applies.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.connectors.base import Connector
from app.rag.ingestion.models import SourceDocument

_SKIP_SUFFIXES = {".acl.json"}


def resolve_local_path(source_uri: str, allowed_root: str | Path) -> Path:
    """Resolve a path relative to `allowed_root` and reject anything that escapes it."""
    root = Path(allowed_root).resolve()
    candidate = Path(source_uri)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise DocumentLoadError(f"Source path must be inside the ingestion root ({root})")
    if not resolved.exists():
        raise DocumentLoadError(f"Source path not found: {source_uri}")
    return resolved


def _read(path: Path) -> SourceDocument:
    stat = path.stat()
    acl_file = path.with_name(path.name + ".acl.json")
    acl = json.loads(acl_file.read_text(encoding="utf-8")) if acl_file.is_file() else []
    return SourceDocument(
        source=str(path),
        name=path.name,
        content=path.read_bytes(),
        acl=[str(a) for a in acl],
        etag=f"{int(stat.st_mtime)}-{stat.st_size}",
        last_modified=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        metadata={"path": str(path)},
    )


class LocalConnector(Connector):
    schemes = ("file://",)

    def __init__(self, root: str | Path) -> None:
        self._root = root

    def source_prefix(self, uri: str) -> str:
        return str(resolve_local_path(uri.removeprefix("file://"), self._root))

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        path = resolve_local_path(uri.removeprefix("file://"), self._root)
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            if any(file.name.endswith(s) for s in _SKIP_SUFFIXES) or file.name.startswith("."):
                continue
            yield await asyncio.to_thread(_read, file)

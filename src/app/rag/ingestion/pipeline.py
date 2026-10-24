"""Ingestion pipeline (A10): the per-document DAG (C-140)

  connector -> raw checks (size, malware, retention)        C-057, C-070, C-071, C-098
            -> land raw                                     C-068
            -> parse (+ attachments recursively)            A5, C-083
            -> clean, governance (quality, sensitivity,      A6
               injection), PII masking
            -> metadata                                     A7
            -> change detection / ACL-only update /         C-143, C-145, C-092, C-093
               duplicate handling
            -> chunk (+ parents)                            A8, S-01
            -> enrich (profile, context, questions,         B1
               summaries)
            -> embed (cached) + index vector records,        A9, C-136, C-137
               prune older versions
            -> graph extraction stage                       G-15
            -> manifest + metrics                           C-148, C-149, C-150
Each step is idempotent (C-141); full syncs propagate deletions (C-144) and the run ends
with count reconciliation (C-139).
"""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core import metrics
from app.core.exceptions import DocumentLoadError, IngestionConfigError
from app.core.logging import get_logger
from app.rag.graph.base import ChunkGraph, GraphStore
from app.rag.graph.extraction import GraphExtractor
from app.rag.ingestion.chunking import ChunkingOptions, apply_semantic_boundaries, chunk_document
from app.rag.ingestion.cleaning import clean_document
from app.rag.ingestion.connectors import ConnectorRegistry
from app.rag.ingestion.connectors.local import LocalConnector
from app.rag.ingestion.enrichment import ChunkExtras, Enricher
from app.rag.ingestion.filetypes import safe_relative_path
from app.rag.ingestion.governance import (
    GovernancePolicy,
    PiiMasker,
    check_parsed,
    check_raw,
    minhash,
)
from app.rag.ingestion.manifest import InMemoryManifest, Manifest, ManifestEntry
from app.rag.ingestion.metadata import PIPELINE_VERSION, build_metadata, load_rules
from app.rag.ingestion.models import Chunk, DocumentMetadata, ParsedDocument, SourceDocument
from app.rag.ingestion.parsing import ParserRouter
from app.rag.ingestion.zones import NullZoneStore, ZoneStore
from app.rag.vectorstore.base import VectorRecord, VectorStore

logger = get_logger(__name__)


def stable_doc_id(collection: str, source: str) -> str:
    """C-072: deterministic document id from collection + source URI."""
    return hashlib.sha256(f"{collection}:{source}".encode()).hexdigest()[:24]


def stable_chunk_id(collection: str, source: str, chunk_index: int) -> str:
    return f"{stable_doc_id(collection, source)}-{chunk_index:04d}"


@dataclass
class LoadedDocument:
    """Already-extracted text (tests, programmatic ingestion)."""

    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    acl: list[str] = field(default_factory=list)

    def as_source(self) -> SourceDocument:
        name = Path(self.source).name or "document.txt"
        if "." not in name:
            name += ".txt"
        return SourceDocument(
            source=self.source,
            name=name,
            content=self.text.encode(),
            acl=list(self.acl),
            metadata=dict(self.metadata),
        )


@dataclass
class IngestionStats:
    run_id: str = ""
    targets: list[str] = field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    unchanged: list[str] = field(default_factory=list)
    acl_updated: list[str] = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)
    quarantined: dict[str, str] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    reconciliation: dict[str, int] = field(default_factory=dict)


@dataclass
class SourceConfig:
    """Where the pipeline may read from (kept for callers building a local-only pipeline)."""

    local_root: str = "./data"
    storage_connection_string: str = ""
    storage_account_url: str = ""
    storage_credential: object | None = None


@dataclass
class PipelineOptions:
    chunking: ChunkingOptions = field(default_factory=ChunkingOptions)
    governance: GovernancePolicy = field(default_factory=GovernancePolicy)
    default_targets: tuple[str, ...] = ("vector",)
    default_acl: tuple[str, ...] = ("*",)
    embedding_model: str = ""
    metadata_rules_path: str = "config/metadata_rules.yaml"
    document_concurrency: int = 4
    graph_concurrency: int = 4
    skip_duplicates: bool = True


@dataclass
class _Run:
    run_id: str
    collection: str
    targets: set[str]
    acl: list[str] | None
    metadata: dict[str, str]
    force: bool
    stats: IngestionStats
    seen: set[str] = field(default_factory=set)


class IngestionPipeline:
    def __init__(
        self,
        embedding_client,
        vector_store: VectorStore | None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        sources: SourceConfig | None = None,
        graph_store: GraphStore | None = None,
        graph_extractor: GraphExtractor | None = None,
        default_targets: Sequence[str] = ("vector",),
        graph_concurrency: int = 4,
        default_acl: Sequence[str] = ("*",),
        embedding_model: str = "",
        *,
        connectors: ConnectorRegistry | None = None,
        parser: ParserRouter | None = None,
        manifest: Manifest | None = None,
        zones: ZoneStore | None = None,
        enricher: Enricher | None = None,
        options: PipelineOptions | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._graph_extractor = graph_extractor
        sources = sources or SourceConfig()
        self._connectors = connectors or ConnectorRegistry(LocalConnector(sources.local_root))
        if connectors is None and (
            sources.storage_connection_string or sources.storage_account_url
        ):
            from app.rag.ingestion.connectors.blob import BlobConnector

            self._connectors.register(
                "blob://",
                lambda: BlobConnector(
                    sources.storage_connection_string,
                    sources.storage_account_url,
                    sources.storage_credential,
                ),
            )
        self._parser = parser or ParserRouter()
        self._manifest = manifest or InMemoryManifest()
        self._zones = zones or NullZoneStore()
        self._enricher = enricher
        self._options = options or PipelineOptions(
            chunking=ChunkingOptions(
                chunk_size=chunk_size, chunk_overlap=min(chunk_overlap, chunk_size // 2)
            ),
            default_targets=tuple(default_targets),
            default_acl=tuple(default_acl),
            embedding_model=embedding_model,
            graph_concurrency=graph_concurrency,
        )
        self._pii = PiiMasker(self._options.governance)
        self._rules = load_rules(self._options.metadata_rules_path)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _resolve_targets(self, targets: Sequence[str] | None) -> set[str]:
        resolved = set(targets or self._options.default_targets)
        if "vector" in resolved and (self._vector_store is None or self._embedding_client is None):
            raise IngestionConfigError("Vector ingestion requested but no vector store configured")
        if "graph" in resolved and (self._graph_store is None or self._graph_extractor is None):
            raise IngestionConfigError("Graph ingestion requested but no graph store configured")
        return resolved

    # -- entry points -----------------------------------------------------------------------
    async def run(
        self,
        source_uri: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        targets: Sequence[str] | None = None,
        acl: Sequence[str] | None = None,
        *,
        full_sync: bool = False,
        force: bool = False,
    ) -> IngestionStats:
        run = self._new_run(collection, targets, acl, metadata, force)
        connector = self._connectors.for_uri(source_uri)
        semaphore = asyncio.Semaphore(self._options.document_concurrency)
        tasks: list[asyncio.Task] = []

        async def guarded(document: SourceDocument) -> None:
            async with semaphore:
                await self._process(document, run)

        async for document in connector.iterate(source_uri):  # C-147: bounded parallelism
            run.seen.add(document.source)
            tasks.append(asyncio.create_task(guarded(document)))
        await asyncio.gather(*tasks)

        if full_sync:  # C-144 / C-146: sources gone from the listing are deleted everywhere
            prefix = connector.source_prefix(source_uri)
            for entry in await self._manifest.list_entries(collection, prefix):
                if entry.source not in run.seen and entry.status not in ("deleted",):
                    await self.delete_source(entry.source, collection)
                    run.stats.deleted.append(entry.source)
        await self._reconcile(run)
        return run.stats

    async def run_for_upload(
        self,
        filename: str,
        data: bytes,
        collection: str,
        metadata: dict[str, str] | None = None,
        targets: Sequence[str] | None = None,
        acl: Sequence[str] | None = None,
        relative_path: str = "",
    ) -> IngestionStats:
        """`relative_path` (folder uploads, e.g. `policies/hr/leave.pdf`) keeps files with the
        same name in different folders apart; it is sanitised and only used in the source id."""
        name = Path(filename).name  # never trust client-supplied directories
        path = safe_relative_path(relative_path, name) if relative_path else name
        folder = path.rpartition("/")[0]
        if folder:
            metadata = {"folder": folder, **(metadata or {})}
        run = self._new_run(collection, targets, acl, metadata, force=True)
        await self._process(
            SourceDocument(source=f"upload://{path}", name=name, content=data),
            run,
            raise_errors=True,
        )
        return run.stats

    async def run_for_documents(
        self,
        documents: list[LoadedDocument | SourceDocument],
        collection: str,
        metadata: dict[str, str] | None = None,
        targets: Sequence[str] | None = None,
        acl: Sequence[str] | None = None,
        force: bool = True,
    ) -> IngestionStats:
        run = self._new_run(collection, targets, acl, metadata, force)
        for document in documents:
            source_doc = document.as_source() if isinstance(document, LoadedDocument) else document
            await self._process(source_doc, run)
        return run.stats

    def _new_run(self, collection, targets, acl, metadata, force) -> _Run:
        resolved = self._resolve_targets(targets)  # fail before loading anything
        run_id = uuid.uuid4().hex[:12]
        return _Run(
            run_id=run_id,
            collection=collection,
            targets=resolved,
            acl=list(acl) if acl else None,
            metadata=dict(metadata or {}),
            force=force,
            stats=IngestionStats(run_id=run_id, targets=sorted(resolved)),
        )

    # -- per document -------------------------------------------------------------------------
    async def _quarantine(
        self, document: SourceDocument, run: _Run, reason: str, doc_id: str
    ) -> None:
        await self._zones.write(
            "quarantine", run.collection, document.source, document.name, document.content
        )
        await self._zones.write("quarantine", run.collection, document.source, "reason.txt", reason)
        await self._manifest.upsert(
            ManifestEntry(
                run.collection,
                document.source,
                doc_id,
                status="quarantined",
                content_hash=document.content_hash,
                etag=document.etag,
                run_id=run.run_id,
                error=reason,
            )
        )
        run.stats.quarantined[document.source] = reason
        metrics.ingest_documents.add(1, {"outcome": "quarantined"})
        logger.warning("document_quarantined", source=document.source, reason=reason)

    async def _process(
        self, document: SourceDocument, run: _Run, raise_errors: bool = False
    ) -> None:
        started = time.perf_counter()
        collection = run.collection
        doc_id = stable_doc_id(collection, document.source)
        try:
            raw_decision = check_raw(document, self._options.governance)
            if raw_decision.action == "quarantine":
                await self._quarantine(document, run, ";".join(raw_decision.reasons), doc_id)
                return
            prior = await self._manifest.get(collection, document.source)
            acl = document.acl or run.acl or list(self._options.default_acl)
            if prior and not run.force and prior.status in ("indexed", "graphed"):
                if (
                    prior.content_hash == document.content_hash
                    and prior.pipeline_version == PIPELINE_VERSION
                ):
                    if sorted(prior.acl) != sorted(acl):  # C-145: permissions changed only
                        await self._update_acl(document.source, collection, acl)
                        prior.acl = acl
                        await self._manifest.upsert(prior)
                        run.stats.acl_updated.append(document.source)
                    else:  # C-143: unchanged
                        run.stats.unchanged.append(document.source)
                    metrics.ingest_documents.add(1, {"outcome": "unchanged"})
                    return

            await self._zones.write(
                "raw", collection, document.source, document.name, document.content
            )
            try:
                parsed = await self._parser.parse(document)
            except DocumentLoadError as exc:
                if raise_errors:
                    raise
                await self._quarantine(document, run, f"parse_failed:{exc.message}", doc_id)
                return
            for attachment in parsed.attachments:  # C-083
                await self._process(attachment, run)
            if not parsed.elements and parsed.attachments:
                return  # an archive / email envelope with only attachments
            clean_document(parsed)
            await self._zones.write(
                "parsed", collection, document.source, "document.md", parsed.to_markdown()
            )

            decision = check_parsed(document, parsed, self._options.governance)
            if decision.action == "quarantine":
                if raise_errors:
                    raise DocumentLoadError(f"Rejected: {';'.join(decision.reasons)}")
                await self._quarantine(document, run, ";".join(decision.reasons), doc_id)
                return
            pii_found = await self._pii.apply(parsed)
            await self._zones.write(
                "clean", collection, document.source, "document.md", parsed.to_markdown()
            )

            signature = minhash(parsed.text)
            if self._options.skip_duplicates and not run.force:
                duplicate = await self._manifest.find_duplicate(
                    collection,
                    document.source,
                    document.content_hash,
                    signature,
                    self._options.governance.near_duplicate_threshold,
                )
                if duplicate is not None and not self._newer(document, duplicate):
                    run.stats.duplicates[document.source] = duplicate.source  # C-092/C-093
                    await self._manifest.upsert(
                        ManifestEntry(
                            collection,
                            document.source,
                            doc_id,
                            status="skipped",
                            content_hash=document.content_hash,
                            duplicate_of=duplicate.source,
                            minhash=signature,
                            run_id=run.run_id,
                        )
                    )
                    metrics.ingest_documents.add(1, {"outcome": "duplicate"})
                    return

            meta = build_metadata(
                document,
                parsed,
                doc_id=doc_id,
                collection=collection,
                acl=acl,
                sensitivity=decision.sensitivity,
                rules=self._rules,
                embedding_model=self._options.embedding_model,
                extra=run.metadata,
            )
            if pii_found:
                meta.extra["pii_masked"] = str(pii_found)
            if decision.reasons:
                meta.extra["governance_flags"] = ";".join(decision.reasons)
            await self._apply_profile(parsed, meta)

            if self._options.chunking.strategy == "semantic" and self._embedding_client is not None:
                await apply_semantic_boundaries(
                    parsed, self._embedding_client, self._options.chunking.semantic_percentile
                )
            chunks = chunk_document(parsed, doc_id, self._options.chunking)
            extras = await self._enricher.enrich_chunks(parsed, chunks) if self._enricher else {}
            await self._zones.write(
                "chunks",
                collection,
                document.source,
                "chunks.json",
                json.dumps([asdict(c) for c in chunks], ensure_ascii=False),
            )

            version = run.run_id
            written = entities = relations = 0
            if "vector" in run.targets:
                written = await self._index_vectors(meta, chunks, extras, version)
            if "graph" in run.targets:
                entities, relations = await self._index_graph(meta, chunks)

            await self._manifest.upsert(
                ManifestEntry(
                    collection,
                    document.source,
                    doc_id,
                    status="graphed" if "graph" in run.targets else "indexed",
                    content_hash=document.content_hash,
                    etag=document.etag,
                    acl=acl,
                    title=meta.title,
                    last_modified=document.last_modified or "",
                    version=version,
                    run_id=run.run_id,
                    pipeline_version=PIPELINE_VERSION,
                    chunk_count=written,
                    entity_count=entities,
                    minhash=signature,
                    reasons=decision.reasons,
                )
            )
            stats = run.stats
            stats.document_count += 1
            stats.chunk_count += written
            stats.entity_count += entities
            stats.relation_count += relations
            metrics.ingest_documents.add(1, {"outcome": "indexed"})
            metrics.ingest_chunks.add(written)
            logger.info(
                "document_ingested",
                source=document.source,
                collection=collection,
                chunks=written,
                entities=entities,
                run_id=run.run_id,
            )
        except IngestionConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
            if raise_errors:
                raise
            logger.exception("document_failed", source=document.source)
            run.stats.failed[document.source] = str(exc)
            await self._manifest.upsert(
                ManifestEntry(
                    collection,
                    document.source,
                    doc_id,
                    status="failed",
                    error=str(exc)[:500],
                    run_id=run.run_id,
                )
            )
            metrics.ingest_documents.add(1, {"outcome": "failed"})
        finally:
            metrics.ingest_duration.record((time.perf_counter() - started) * 1000)

    @staticmethod
    def _newer(document: SourceDocument, existing: ManifestEntry) -> bool:
        """C-093: a near-duplicate is indexed (as the canonical version) only when its source
        is newer than the copy already indexed; exact copies are always skipped."""
        if document.content_hash == existing.content_hash:
            return False
        return bool(
            document.last_modified
            and existing.last_modified
            and document.last_modified > existing.last_modified
        )

    async def _apply_profile(self, parsed: ParsedDocument, meta: DocumentMetadata) -> None:
        if self._enricher is None:
            return
        profile = await self._enricher.profile(parsed)
        if profile.get("summary"):
            meta.summary = str(profile["summary"])
        if profile.get("doc_type") and meta.doc_type == "other":
            meta.doc_type = str(profile["doc_type"])
        if profile.get("effective_date") and not meta.effective_date:
            meta.effective_date = str(profile["effective_date"])[:10]
        if profile.get("tags"):
            meta.tags = sorted({*meta.tags, *(str(t) for t in profile["tags"][:5])})

    # -- stores --------------------------------------------------------------------------------
    def _record(
        self, meta: DocumentMetadata, chunk: Chunk, version: str, **overrides
    ) -> VectorRecord:
        values = dict(
            chunk_id=chunk.chunk_id,
            content=chunk.text,
            embedding=None,
            source=meta.source,
            collection=meta.collection,
            chunk_index=chunk.index,
            content_hash=chunk.content_hash,
            metadata={
                **meta.chunk_metadata(),
                **({"context": chunk.context} if chunk.context else {}),
                "pipeline_version": PIPELINE_VERSION,
            },
            doc_id=meta.doc_id,
            title=meta.title,
            section_path=" > ".join(chunk.section_path),
            acl=meta.acl,
            doc_type=meta.doc_type,
            language=meta.language,
            tags=meta.tags,
            effective_date=meta.effective_date,
            page=chunk.page_start,
            parent_id=chunk.parent_id,
            prev_id=chunk.prev_id,
            next_id=chunk.next_id,
            kind=chunk.kind,
            embedding_model=self._options.embedding_model,
            ingest_version=version,
        )
        values.update(overrides)
        return VectorRecord(**values)

    async def _index_vectors(
        self,
        meta: DocumentMetadata,
        chunks: list[Chunk],
        extras: dict[str, ChunkExtras],
        version: str,
    ) -> int:
        records = [self._record(meta, c, version) for c in chunks]
        for chunk in chunks:  # S-06 / S-07 multi-vector representations
            extra = extras.get(chunk.chunk_id)
            if extra is None:
                continue
            for i, question in enumerate(extra.questions):
                records.append(
                    self._record(
                        meta,
                        chunk,
                        version,
                        chunk_id=f"{chunk.chunk_id}-q{i}",
                        content=question,
                        kind="question",
                        target_id=chunk.chunk_id,
                        prev_id="",
                        next_id="",
                    )
                )
            if extra.summary:
                records.append(
                    self._record(
                        meta,
                        chunk,
                        version,
                        chunk_id=f"{chunk.chunk_id}-s",
                        content=extra.summary,
                        kind="summary",
                        target_id=chunk.chunk_id,
                        prev_id="",
                        next_id="",
                    )
                )
        if meta.summary:  # S-14: document-level summary for coarse retrieval
            first = next((c for c in chunks if c.kind == "chunk"), None)
            if first is not None:
                records.append(
                    self._record(
                        meta,
                        first,
                        version,
                        chunk_id=f"{meta.doc_id}-docsum",
                        content=meta.summary,
                        kind="summary",
                        target_id=first.chunk_id,
                        prev_id="",
                        next_id="",
                    )
                )
        to_embed = [r for r in records if r.kind != "parent"]
        vectors = await self._embedding_client.embed_batch([r.content for r in to_embed])
        for record, vector in zip(to_embed, vectors, strict=True):
            record.embedding = vector
        written = await self._vector_store.upsert(records)
        # Older versions of this document are removed only after the new ones exist (C-137).
        await self._vector_store.delete_by_source(
            meta.source,
            meta.collection,
            except_version=version,
            keep_ids={r.chunk_id for r in records},
        )
        return sum(1 for r in records if r.kind == "chunk") if written else 0

    async def _update_acl(self, source: str, collection: str, acl: list[str]) -> None:
        if self._vector_store is not None:
            await self._vector_store.update_acl(source, collection, acl)
        if self._graph_store is not None and hasattr(self._graph_store, "update_acl"):
            await self._graph_store.update_acl(source, collection, acl)

    async def _index_graph(self, meta: DocumentMetadata, chunks: list[Chunk]) -> tuple[int, int]:
        extractor = self._graph_extractor
        assert extractor is not None and self._graph_store is not None  # noqa: S101
        await self._graph_store.delete_by_source(meta.source, meta.collection)
        semaphore = asyncio.Semaphore(self._options.graph_concurrency)

        async def extract(chunk: Chunk) -> ChunkGraph:
            async with semaphore:
                entities, relations, claims = await extractor.extract_all(
                    chunk.raw_text or chunk.text
                )
            return ChunkGraph(
                chunk_id=chunk.chunk_id,
                source=meta.source,
                collection=meta.collection,
                content=chunk.text,
                chunk_index=chunk.index,
                acl=meta.acl,
                doc_id=meta.doc_id,
                title=meta.title,
                section_path=chunk.section_path,  # lexical layer (G-20)
                prev_id=chunk.prev_id,
                entities=entities,
                relations=relations,
                claims=claims,
            )

        graphs = await asyncio.gather(*(extract(c) for c in chunks if c.kind == "chunk"))
        for chunk_graph in graphs:  # sequential writes keep MERGEs race-free
            await self._graph_store.upsert_chunk_graph(chunk_graph)
        await self._embed_entities(meta.collection, graphs)
        return sum(len(g.entities) for g in graphs), sum(len(g.relations) for g in graphs)

    async def _embed_entities(self, collection: str, graphs: list[ChunkGraph]) -> None:
        """G-25: embed new/updated entity descriptions so questions can link to them by meaning."""
        if self._embedding_client is None:
            return
        entities = {e.key: e for g in graphs for e in g.entities}
        if not entities:
            return
        texts = {k: f"{e.name} ({e.type}): {e.description}".strip() for k, e in entities.items()}
        vectors = await self._embedding_client.embed_batch(list(texts.values()))
        await self._graph_store.upsert_entity_embeddings(
            collection, [(k, t, v) for (k, t), v in zip(texts.items(), vectors, strict=True)]
        )

    async def delete_source(self, source: str, collection: str) -> tuple[int, int]:
        chunks = (
            await self._vector_store.delete_by_source(source, collection)
            if self._vector_store
            else 0
        )
        nodes = (
            await self._graph_store.delete_by_source(source, collection) if self._graph_store else 0
        )
        entry = await self._manifest.get(collection, source)
        if entry is not None:
            entry.status = "deleted"
            await self._manifest.upsert(entry)
        return chunks, nodes

    async def _reconcile(self, run: _Run) -> None:
        """C-139: index record counts per touched source must match what the run wrote."""
        if self._vector_store is None or "vector" not in run.targets:
            return
        mismatches = 0
        for source in run.seen:
            entry = await self._manifest.get(run.collection, source)
            if (
                entry is None
                or entry.status not in ("indexed", "graphed")
                or entry.run_id != run.run_id
            ):
                continue
            stored = await self._vector_store.count(run.collection, source)
            if stored < entry.chunk_count:
                mismatches += 1
                logger.warning(
                    "reconciliation_mismatch",
                    source=source,
                    expected=entry.chunk_count,
                    stored=stored,
                )
        run.stats.reconciliation = {"checked": len(run.seen), "mismatches": mismatches}

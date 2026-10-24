"""Builds the ingestion pipeline's parts from Settings (connectors, parsers, governance,
enrichment, manifest, zones). Anything whose placeholder settings are blank is skipped."""

from functools import lru_cache
from pathlib import Path

import yaml

from app.core.azure_auth import get_azure_credential
from app.core.config import Settings
from app.rag.ingestion.chunking import ChunkingOptions
from app.rag.ingestion.connectors import ConnectorRegistry
from app.rag.ingestion.connectors.local import LocalConnector
from app.rag.ingestion.enrichment import Enricher, EnrichmentOptions
from app.rag.ingestion.governance import GovernancePolicy
from app.rag.ingestion.manifest import InMemoryManifest, Manifest, MongoManifest
from app.rag.ingestion.parsing import ParserRouter
from app.rag.ingestion.pipeline import PipelineOptions
from app.rag.ingestion.zones import BlobZoneStore, LocalZoneStore, NullZoneStore, ZoneStore


@lru_cache(maxsize=2)
def _acl_mapping(path: str) -> dict[str, dict[str, str]]:
    file = Path(path)
    if not file.is_file():
        return {}
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return {k: {str(a): str(b) for a, b in (v or {}).items()} for k, v in data.items()}


def build_connectors(settings: Settings) -> ConnectorRegistry:
    registry = ConnectorRegistry(LocalConnector(settings.ingestion_local_root))
    storage_secret = settings.azure_storage_connection_string.get_secret_value()
    if storage_secret or settings.azure_storage_account_url:
        from app.rag.ingestion.connectors.blob import BlobConnector

        registry.register(
            "blob://",
            lambda: BlobConnector(
                storage_secret,
                settings.azure_storage_account_url,
                None if storage_secret else get_azure_credential(),
            ),
        )
    if settings.sharepoint_client_id or settings.sharepoint_use_managed_identity:
        from app.rag.ingestion.connectors.sharepoint import SharePointConnector

        registry.register(
            "sharepoint://",
            lambda: SharePointConnector(
                tenant_id=settings.sharepoint_tenant_id,
                client_id=settings.sharepoint_client_id,
                client_secret=settings.sharepoint_client_secret.get_secret_value(),
                credential=get_azure_credential(),
                site_group_map=_acl_mapping(settings.acl_mapping_path).get(
                    "sharepoint_site_groups", {}
                ),
            ),
        )
    if settings.confluence_base_url:
        from app.rag.ingestion.connectors.confluence import ConfluenceConnector

        registry.register(
            "confluence://",
            lambda: ConfluenceConnector(
                settings.confluence_base_url,
                settings.confluence_user,
                settings.confluence_api_token.get_secret_value(),
                group_map=_acl_mapping(settings.acl_mapping_path).get("confluence_groups", {}),
                default_acl=settings.confluence_default_acl,
            ),
        )
    if Path(settings.sources_config_path).is_file():
        from app.rag.ingestion.connectors.database import DatabaseConnector
        from app.rag.ingestion.connectors.rest_api import RestApiConnector

        registry.register("db://", lambda: DatabaseConnector(settings.sources_config_path))
        registry.register("api://", lambda: RestApiConnector(settings.sources_config_path))
    if settings.web_crawl_enabled:
        from app.rag.ingestion.connectors.web import WebConnector

        def web() -> WebConnector:
            return WebConnector(
                max_depth=settings.web_crawl_max_depth,
                max_pages=settings.web_crawl_max_pages,
                delay_seconds=settings.web_crawl_delay_seconds,
                auth_header=settings.web_crawl_auth_header.get_secret_value(),
                acl=settings.ingestion_default_acl,
            )

        registry.register("https://", web)
        registry.register("http://", web)
    return registry


def build_parser(settings: Settings) -> ParserRouter:
    docintel = vision = transcriber = None
    uses_docintel = settings.pdf_parser != "native" or settings.image_parser in (
        "auto",
        "docintel",
        "docintel+vision",
    )
    if settings.doc_intelligence_endpoint and uses_docintel:
        from app.rag.ingestion.docintel import DocumentIntelligenceParser

        docintel = DocumentIntelligenceParser(
            settings.doc_intelligence_endpoint,
            settings.doc_intelligence_api_key.get_secret_value(),
            image_model=settings.doc_intelligence_image_model,
        )
    openai_args = (
        settings.azure_openai_endpoint,
        settings.azure_openai_api_key.get_secret_value(),
        settings.azure_openai_api_version,
    )
    if settings.azure_openai_vision_deployment and settings.azure_openai_endpoint:
        from app.rag.ingestion.media import VisionDescriber

        vision = VisionDescriber(
            *openai_args,
            settings.azure_openai_vision_deployment,
            max_tokens=settings.vision_max_tokens,
        )
    if settings.azure_openai_whisper_deployment and settings.azure_openai_endpoint:
        from app.rag.ingestion.media import Transcriber

        transcriber = Transcriber(*openai_args, settings.azure_openai_whisper_deployment)
    return ParserRouter(
        docintel,
        vision,
        transcriber,
        pdf_parser=settings.pdf_parser,
        image_parser=settings.image_parser,
        image_min_ocr_chars=settings.image_min_ocr_chars,
        pdf_image_descriptions=settings.pdf_image_descriptions,
        pdf_image_min_pixels=settings.pdf_image_min_pixels,
        pdf_max_images=settings.pdf_max_images,
        vision_concurrency=settings.vision_concurrency,
    )


def build_manifest(settings: Settings) -> Manifest:
    if settings.manifest_store == "mongo":
        from app.core.mongo import get_mongo_database

        database = get_mongo_database(
            settings.mongodb_url.get_secret_value(), settings.mongodb_database
        )
        return MongoManifest(database[settings.manifest_collection])
    return InMemoryManifest()


def build_zones(settings: Settings) -> ZoneStore:
    if settings.zones_store == "local":
        return LocalZoneStore(settings.zones_local_root)
    if settings.zones_store == "blob":
        return BlobZoneStore(
            settings.data_lake_account_url, settings.data_lake_container, get_azure_credential()
        )
    return NullZoneStore()


def build_enricher(settings: Settings, llm) -> Enricher | None:
    options = EnrichmentOptions(
        profile=settings.enrich_profile,
        context=settings.enrich_contextual,
        questions=settings.enrich_questions,
        summaries=settings.enrich_summaries,
        doc_types=settings.doc_types,
    )
    enricher = Enricher(llm, options)
    return enricher if enricher.enabled else None


def build_options(settings: Settings) -> PipelineOptions:
    return PipelineOptions(
        chunking=ChunkingOptions(
            chunk_size=settings.chunk_size,
            chunk_overlap=min(settings.chunk_overlap, settings.chunk_size // 2),
            min_chunk_tokens=settings.chunk_min_tokens,
            parent_size=settings.chunk_parent_size,
            prepend_header=settings.chunk_prepend_header,
            strategy=settings.chunk_strategy,
        ),
        governance=GovernancePolicy(
            max_bytes=settings.ingestion_max_file_mb * 1024 * 1024,
            min_text_chars=settings.ingestion_min_text_chars,
            min_parse_quality=settings.ingestion_min_parse_quality,
            max_sensitivity=settings.ingestion_max_sensitivity,
            injection_action=settings.ingestion_injection_action,
            pii_mode=settings.pii_mode,
            pii_action=settings.pii_action,
            pii_endpoint=settings.language_endpoint,
            pii_api_key=settings.language_api_key.get_secret_value(),
            retention_max_age_days=settings.retention_max_age_days,
            near_duplicate_threshold=settings.near_duplicate_threshold,
        ),
        default_targets=tuple(settings.ingestion_targets),
        default_acl=tuple(settings.ingestion_default_acl),
        embedding_model=(
            f"{settings.azure_openai_embedding_deployment}@{settings.embedding_dimensions}"
        ),
        metadata_rules_path=settings.metadata_rules_path,
        document_concurrency=settings.ingestion_document_concurrency,
    )

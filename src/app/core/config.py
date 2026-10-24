"""Application settings.

Where each value comes from (highest priority first):
  1. Environment variables / App Service app settings — URLs, flags, tuning (non-secret).
  2. Azure Key Vault (when AZURE_KEY_VAULT_URL is set) — secrets. A secret named
     `azure-openai-api-key` fills AZURE_OPENAI_API_KEY (dashes <-> underscores).
  3. `.env` file — local development only (never committed).
  4. Defaults below.
Empty values are ignored at every level, so a blank `KEY=` never masks a Key Vault secret.
"""

import json
import os
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class RagMode(StrEnum):
    SEMANTIC = "semantic"  # dense vector search over chunks
    HYBRID = "hybrid"  # dense vector + keyword (BM25) search, fused with RRF
    HYBRID_GRAPH = "hybrid_graph"  # hybrid search + knowledge-graph facts, fused with RRF
    GRAPH = "graph"  # entity + relationship traversal (MongoDB graph collections)
    AGENTIC = "agentic"  # LangGraph agent: route -> retrieve -> grade -> rewrite loop
    AUTO = "auto"  # unified router: pick per question by complexity (I-01)


IngestionTarget = Literal["vector", "graph"]
VectorProvider = Literal["mongo"]


def _url(value: str) -> str:
    """Optional URL: empty is allowed (feature off), otherwise must be http(s)."""
    value = value.strip()
    if value and not value.startswith(("https://", "http://")):
        raise ValueError(f"must be an http(s) URL, got {value!r}")
    return value.rstrip("/")


Url = Annotated[str, AfterValidator(_url)]


def _csv_list(value: object) -> object:
    """Accept `a,b` as well as JSON `["a","b"]` for list settings in env vars."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            return json.loads(value)
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Environment-driven application configuration (12-factor style)."""

    model_config = SettingsConfigDict(
        # APP_ENV_FILE lets tests (or a second local profile) point elsewhere.
        env_file=os.getenv("APP_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # --- App -------------------------------------------------------------------------
    app_env: str = "local"
    app_name: str = "oissa-rag-agent"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Built-in chat page at /ui (C-230). Turn off when a separate front end is used.
    ui_enabled: bool = True
    # Organisation name used in prompts ("knowledge assistant for <organisation>").
    organisation_name: str = "the organisation"  # TODO: set, e.g. "the World Bank Group"
    # One-line description of what the assistant covers (used for out-of-scope detection).
    assistant_scope: str = ""  # TODO: e.g. "HR, finance and procurement policies"

    # --- Authentication & access control (C-048, C-154, C-192) ---------------------------
    # none | api_key | entra | entra+api_key  (none = local only; ACLs not enforced)
    auth_mode: Literal["none", "api_key", "entra", "entra+api_key"] = "none"
    api_key: SecretStr = SecretStr("")  # Key Vault secret: api-key
    # Principals granted to API-key callers ("*" = documents shared with everyone).
    api_key_principals: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    entra_tenant_id: str = ""  # TODO: Entra tenant (directory) id
    # Accepted token audiences: the API app's client id and/or App ID URI (api://...).
    entra_api_audience: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Entra app role required to ingest/delete content (blank = any authenticated caller).
    ingestion_required_role: str = ""  # TODO: e.g. "Ingestion.Write"
    # Entra app role required for the admin RAG studio (/admin/*: model and mode comparison,
    # evaluation). Blank = any authenticated caller; API-key and local callers are trusted.
    admin_required_role: str = "SA_Admin"
    # Enforce document ACLs in every retriever (disable only for single-audience corpora).
    acl_enforcement: bool = True
    # ACL applied to ingested documents that carry none ("*" = everyone in the org).
    ingestion_default_acl: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    # Requests per minute per user / API key (0 = unlimited). Shared across instances via Redis.
    rate_limit_per_minute: int = 60
    redis_url: SecretStr = SecretStr("")  # rediss://:<key>@<name>.redis.cache.windows.net:6380

    # --- Guardrails (C-156, C-194, C-195) -------------------------------------------------
    content_safety_mode: Literal["off", "heuristic", "azure"] = "heuristic"
    content_safety_endpoint: Url = ""  # https://<name>.cognitiveservices.azure.com
    content_safety_api_key: SecretStr = SecretStr("")  # blank = managed identity
    content_safety_severity_threshold: int = Field(default=4, ge=0, le=7)
    block_output_pii: bool = True
    log_pii_masking: bool = True

    # --- Secrets source ----------------------------------------------------------------
    azure_key_vault_url: Url = ""  # e.g. https://kv-oissarag-qa.vault.azure.net

    # --- RAG mode switch -----------------------------------------------------------------
    # Default strategy for requests that don't name one.
    rag_mode: RagMode = RagMode.SEMANTIC
    # Strategies this deployment initialises and accepts (others are rejected with 400).
    rag_enabled_modes: Annotated[list[RagMode], NoDecode] = Field(
        default_factory=lambda: [RagMode.SEMANTIC]
    )
    # Let callers pick `rag_mode` per request (side-by-side validation). Disable in prod
    # once a strategy is chosen.
    rag_allow_mode_override: bool = True
    # Include the retrieval trace (steps, queries, timings) in responses when requested.
    rag_allow_trace: bool = True
    # Unified router (RAG_MODE=auto): question complexity -> mode, JSON (I-01).
    rag_router_map: dict[str, RagMode] = Field(
        default_factory=lambda: {
            "simple": RagMode.HYBRID,
            "relationship": RagMode.GRAPH,
            "global": RagMode.GRAPH,
            "multi_step": RagMode.AGENTIC,
        }
    )
    # I-03: retry a low-confidence answer with the next more capable enabled mode.
    rag_escalation_enabled: bool = False
    rag_max_escalations: int = Field(default=1, ge=0, le=2)

    # --- Observability -------------------------------------------------------------------
    applicationinsights_connection_string: SecretStr = SecretStr("")
    # Log the user's question text with each answer (off by default: may contain PII).
    log_question_text: bool = False

    # --- Answer quality ----------------------------------------------------------------------
    # LLM judge (groundedness / answer relevance / context relevance, 1-5):
    #   off      never
    #   request  when the request sets "evaluate": true
    #   always   on every chat answer (adds one LLM call of latency and cost)
    quality_judge: Literal["off", "request", "always"] = "request"
    # Share of other answers judged in the background for dashboards (0.0-1.0).
    quality_judge_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Characters of each passage the judge reads; keep >= the longest context (parent sections)
    quality_judge_max_passage_chars: int = Field(default=8000, ge=500, le=50000)

    # --- Azure OpenAI (empty key => Managed Identity / az login) -------------------------
    azure_openai_endpoint: Url = ""  # https://<name>.openai.azure.com
    azure_openai_api_key: SecretStr = SecretStr("")  # Key Vault: azure-openai-api-key
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o"
    # Cheaper model for routing/grading/extraction; falls back to the chat deployment.
    azure_openai_utility_deployment: str = ""
    # Answer deployments the admin playground may pick (comma-separated; same Azure OpenAI
    # resource). Blank = the chat, utility and fallback deployments.
    rag_selectable_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 64
    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # --- MongoDB Atlas: chunks (vector + keyword search), knowledge graph, manifest --------
    # mongodb+srv://user:pass@cluster/... (Key Vault: mongodb-url). Vector and keyword search
    # use Atlas Search indexes, created on startup (MONGODB_CREATE_INDEXES_ON_STARTUP).
    mongodb_url: SecretStr = SecretStr("")
    mongodb_database: str = Field(default="knowledge", pattern=r"^[A-Za-z0-9_\-]{1,63}$")
    mongodb_create_indexes_on_startup: bool = True
    # Default store for reads (and the only one when VECTOR_STORE_PROVIDERS is blank).
    vector_store_provider: VectorProvider = "mongo"
    # Stores to keep in sync: ingestion writes to all of them and a request's `vector_store`
    # picks which one answers (compare quality and latency).
    vector_store_providers: Annotated[list[VectorProvider], NoDecode] = Field(default_factory=list)
    retrieval_mode: str = Field(default="hybrid", pattern="^(vector|hybrid|semantic_hybrid)$")
    mongodb_chunks_collection: str = "rag_chunks"
    mongodb_vector_index: str = "rag_chunks_vector"  # Atlas Vector Search index
    mongodb_search_index: str = "rag_chunks_text"  # Atlas Search (BM25) index
    mongodb_text_analyzer: str = "lucene.english"
    # $vectorSearch numCandidates = top_k x factor (ANN recall vs latency)
    mongodb_num_candidates_factor: int = Field(default=10, ge=1, le=100)
    # RETRIEVAL_MODE=semantic_hybrid: llm | cohere (RERANKER_* endpoint) | none
    mongodb_semantic_reranker: Literal["llm", "cohere", "none"] = "llm"
    mongodb_semantic_candidates: int = Field(default=20, ge=1, le=100)

    # --- Graph RAG: graph collections in the MongoDB database above -----------------------
    graph_store_provider: Literal["mongo"] = "mongo"
    # Collections <prefix>_chunks / _entities / _relations / _claims / _sections / _documents
    mongodb_graph_prefix: str = Field(
        default="rag_graph", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$"
    )
    mongodb_entity_vector_index: str = "rag_graph_entities_vector"  # entity linking (G-25)
    graph_create_schema_on_startup: bool = True
    graph_max_hops: int = Field(default=2, ge=1, le=3)
    graph_max_entities: int = 10
    graph_max_relations: int = 40
    # Search mode: auto | local | path | hybrid | global | drift | dual | cypher (C6)
    graph_search_mode: Literal[
        "auto", "local", "path", "hybrid", "global", "drift", "dual", "cypher"
    ] = "auto"
    graph_per_entity_limit: int = 15  # fan-out cap (G-40)
    graph_min_relation_weight: float = 0.0
    graph_use_ppr: bool = True  # personalized PageRank ranking (G-42)
    graph_knee_cutoff: bool = False  # knee cutoff on PageRank fact / community report scores
    graph_include_claims: bool = True
    graph_global_reports: int = 8
    graph_allow_freeform_cypher: bool = False  # only safe without per-user ACLs (G-44)
    graph_semantic_fallback: bool = True  # G-55
    # Extraction (C2) and resolution (C3)
    graph_ontology_path: str = "config/ontology.yaml"
    graph_aliases_path: str = "config/entity_aliases.yaml"
    graph_cypher_templates_path: str = "config/cypher_templates.yaml"
    graph_gleaning_rounds: int = Field(default=0, ge=0, le=2)  # G-13
    graph_extract_claims: bool = False  # G-12
    # Communities (C4): built from facts visible to this ACL scope only (see communities.py)
    graph_community_acl_scope: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    graph_community_levels: int = 3
    graph_community_min_size: int = 3

    # --- Agentic RAG: LangGraph ----------------------------------------------------------
    # Tools the agent may use (D2): vector (semantic_search), keyword, fetch_document,
    # graph (graph_local_search), graph_global, graph_query, calculator, date_math, web.
    agentic_tools: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "vector",
            "keyword",
            "fetch_document",
            "graph",
            "graph_global",
            "calculator",
            "date_math",
        ]
    )
    agentic_max_iterations: int = Field(default=3, ge=1, le=6)
    agentic_min_relevant: int = 2
    agentic_max_tool_calls: int = 12  # A-34
    agentic_max_tokens: int = 60_000  # A-35 per run (0 = unlimited)
    agentic_step_timeout_seconds: float = 20.0  # A-36
    agentic_max_seconds: float = 90.0
    agentic_max_reflections: int = 1  # A-31
    agentic_allow_clarify: bool = False  # A-45
    agentic_generate_and_check: bool = True  # A-28..A-30
    agentic_review_on_failed_check: bool = True  # A-44
    agentic_checkpointer: Literal["none", "memory", "mongo"] = "none"  # A-06
    # mongo: langgraph-checkpoint-mongodb; blank = MONGODB_URL
    agentic_checkpoint_dsn: SecretStr = SecretStr("")
    # A-13 external web search (Tavily-compatible API). Off unless both are set AND "web" is a tool.
    web_search_endpoint: Url = ""
    web_search_api_key: SecretStr = SecretStr("")

    # --- Source documents ----------------------------------------------------------------
    azure_storage_connection_string: SecretStr = SecretStr("")  # Key Vault (optional)
    azure_storage_account_url: Url = ""  # https://<account>.blob.core.windows.net
    azure_storage_container: str = "documents"
    ingestion_local_root: str = "./data"
    ingestion_max_upload_mb: int = 50
    # File types the upload endpoint / RAG Studio accept (checked by extension and content).
    # Any of the parser's SUPPORTED_EXTENSIONS may be added.
    ingestion_upload_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".xlsx", ".txt", ".json", ".xml", ".png", ".jpg", ".jpeg",
        ]
    )  # fmt: skip
    # Which stores ingestion writes to by default ("graph" runs LLM entity extraction).
    ingestion_targets: Annotated[list[IngestionTarget], NoDecode] = Field(
        default_factory=lambda: ["vector"]
    )
    ingestion_document_concurrency: int = 4
    ingestion_default_collection: str = "default"
    # Queue-driven ingestion (scripts/ingestion_worker.py). Blank = API/CLI ingestion only.
    servicebus_namespace: str = ""  # TODO: <name>.servicebus.windows.net
    servicebus_connection_string: SecretStr = SecretStr("")  # alternative to managed identity
    ingestion_queue_name: str = "ingestion"

    # --- Connectors (A4) — blank = connector disabled ---------------------------------------
    sources_config_path: str = "config/sources.yaml"  # db:// and api:// definitions
    acl_mapping_path: str = "config/acl_mapping.yaml"
    sharepoint_tenant_id: str = ""  # TODO
    sharepoint_client_id: str = ""  # TODO: app registration with Sites.Selected
    sharepoint_client_secret: SecretStr = SecretStr("")  # Key Vault: sharepoint-client-secret
    sharepoint_use_managed_identity: bool = False
    confluence_base_url: Url = ""  # TODO: https://<org>.atlassian.net/wiki
    confluence_user: str = ""
    confluence_api_token: SecretStr = SecretStr("")  # Key Vault: confluence-api-token
    confluence_default_acl: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    web_crawl_enabled: bool = False
    web_crawl_max_depth: int = 2
    web_crawl_max_pages: int = 200
    web_crawl_delay_seconds: float = 1.0
    web_crawl_auth_header: SecretStr = SecretStr("")  # e.g. "Authorization: Bearer ..."

    # --- Parsing (A5) ------------------------------------------------------------------------
    pdf_parser: Literal["auto", "native", "docintel"] = "auto"  # auto = docintel if configured
    doc_intelligence_endpoint: Url = ""  # TODO: https://<name>.cognitiveservices.azure.com
    doc_intelligence_api_key: SecretStr = SecretStr("")  # blank = managed identity
    # Images (C-075, C-078). Document Intelligence reads text (OCR) from PNG, JPEG, BMP, TIFF and
    # HEIF; the vision deployment describes PNG, JPEG, GIF and WebP (photos, diagrams, charts).
    #   auto             OCR when supported; add a vision description when OCR finds little text
    #                    or the format is vision-only
    #   docintel         OCR only        vision    description only
    #   docintel+vision  both            off       images are not ingested
    image_parser: Literal["auto", "docintel", "vision", "docintel+vision", "off"] = "auto"
    # TODO: prebuilt-read (OCR text only, cheaper) or prebuilt-layout (also tables, headings,
    # figures). PDFs always use prebuilt-layout.
    doc_intelligence_image_model: str = "prebuilt-layout"
    # Below this many OCR characters an image is treated as a picture rather than a scan.
    image_min_ocr_chars: int = Field(default=30, ge=0)
    azure_openai_vision_deployment: str = ""  # describe images/figures (C-078), e.g. gpt-4o
    vision_max_tokens: int = Field(default=1500, ge=200, le=8000)  # per image description
    # PDFs parsed natively (no Document Intelligence): describe their embedded images with the
    # vision deployment - diagrams and charts on text pages, and image-only (scanned) pages.
    pdf_image_descriptions: bool = True
    pdf_image_min_pixels: int = Field(default=150, ge=16)  # shorter side; skips icons and logos
    pdf_max_images: int = Field(default=30, ge=0)  # per document; 0 = none
    vision_concurrency: int = Field(default=4, ge=1, le=16)  # images described at once
    azure_openai_whisper_deployment: str = ""  # transcribe audio/video (C-082)

    # --- Cleaning & governance (A6) --------------------------------------------------------------
    ingestion_max_file_mb: int = 100
    ingestion_min_text_chars: int = 50
    ingestion_min_parse_quality: float = 0.4
    ingestion_max_sensitivity: Literal["public", "internal", "confidential", "restricted"] = (
        "confidential"
    )
    ingestion_injection_action: Literal["quarantine", "flag"] = "quarantine"
    pii_mode: Literal["off", "regex", "azure_language"] = "off"
    pii_action: Literal["mask", "flag"] = "mask"
    language_endpoint: Url = (
        ""  # Azure AI Language (PII): https://<name>.cognitiveservices.azure.com
    )
    language_api_key: SecretStr = SecretStr("")
    retention_max_age_days: int = 0  # 0 = keep everything
    near_duplicate_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    metadata_rules_path: str = "config/metadata_rules.yaml"

    # --- Chunking (A8) & enrichment (B1) ----------------------------------------------------------
    chunk_min_tokens: int = 40
    chunk_parent_size: int = 0  # > 0 = parent-child chunking (S-01), e.g. 1500
    chunk_prepend_header: bool = True
    chunk_strategy: Literal["structure", "semantic"] = "structure"  # semantic = S-02
    enrich_profile: bool = False  # doc summary / type / tags (C-104, C-105)
    enrich_contextual: bool = False  # contextual retrieval (S-04)
    enrich_questions: int = 0  # generated questions per chunk (S-06)
    enrich_summaries: bool = False  # chunk summaries (S-07)
    embedding_cache_enabled: bool = True  # C-127

    # --- Ingestion bookkeeping (A10) -------------------------------------------------------------
    manifest_store: Literal["memory", "mongo"] = "memory"
    manifest_collection: str = "rag_ingestion_manifest"  # in MONGODB_DATABASE
    zones_store: Literal["off", "local", "blob"] = "local"
    zones_local_root: str = ".ragdata"
    data_lake_account_url: Url = ""  # TODO: https://<account>.blob.core.windows.net
    data_lake_container: str = "rag-zones"

    # --- Retrieval / chunking tuning -----------------------------------------------------
    retrieval_top_k: int = 8
    retrieval_rerank_top_n: int = 4
    retrieval_min_score: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Query understanding (A11) ------------------------------------------------------------
    # One utility-model call per question: follow-up rewrite, intent, complexity, filters.
    query_analysis_enabled: bool = True
    glossary_path: str = "config/glossary.yaml"
    # Document types the self-query filter may use (keep in sync with ingestion doc_type).
    doc_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "policy",
            "procedure",
            "guideline",
            "contract",
            "faq",
            "manual",
            "report",
            "other",
        ]
    )

    # --- Semantic retrieval extras (Part B) ---------------------------------------------------
    semantic_query_transform: Literal["none", "multi_query", "hyde", "step_back"] = "none"
    semantic_multi_query_count: int = 3
    semantic_mmr: bool = False
    semantic_mmr_lambda: float = 0.7
    semantic_expand: Literal["none", "parent", "window"] = "none"
    semantic_recency_boost: float = 0.0  # e.g. 0.3 = newest docs score up to 30% higher
    semantic_recency_half_life_days: int = 365
    semantic_hierarchical: bool = False  # S-26: documents first (needs ENRICH_PROFILE summaries)
    semantic_hierarchical_docs: int = 5

    # --- Reranking & context assembly (A12) ---------------------------------------------------
    reranker: Literal["none", "llm", "cohere"] = "none"
    reranker_endpoint: Url = ""  # Cohere rerank endpoint (Azure AI Foundry serverless)
    reranker_api_key: SecretStr = SecretStr("")
    reranker_model: str = "rerank-v3.5"
    context_max_tokens: int = 6000
    context_min_relative_score: float = 0.0
    context_reorder: bool = True
    # Knee-point cutoff (Kneedle) on the reranked scores: keep results before the sharp drop
    retrieval_knee_cutoff: bool = False
    retrieval_knee_sensitivity: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Generation (A13) ---------------------------------------------------------------------
    # Prompt version per prompt name, e.g. {"rag_answer": "v2"} (JSON).
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    # Secondary model for failover (C-190): another region / resource. Blank = no failover.
    azure_openai_fallback_endpoint: Url = ""
    azure_openai_fallback_api_key: SecretStr = SecretStr("")
    azure_openai_fallback_chat_deployment: str = ""

    # --- Conversations (C-157, C-191) ---------------------------------------------------------
    conversation_store: Literal["memory", "cosmos"] = "memory"
    conversation_max_turns: int = 6
    cosmos_endpoint: Url = ""  # https://<account>.documents.azure.com:443/
    cosmos_key: SecretStr = SecretStr("")  # blank = managed identity
    cosmos_database: str = "rag"
    cosmos_conversation_container: str = "conversations"

    # --- Answer cache (C-165) -----------------------------------------------------------------
    answer_cache_enabled: bool = False
    answer_cache_similarity: float = Field(default=0.95, ge=0.5, le=1.0)
    answer_cache_ttl_seconds: int = 3600

    # --- Evaluation (admin RAG studio) --------------------------------------------------------
    eval_golden_set_path: str = "evals/golden_set.jsonl"
    eval_thresholds_path: str = "evals/thresholds.yaml"

    # Allow comma-separated lists in env vars / App Service settings.
    @field_validator(
        "cors_allow_origins",
        "rag_enabled_modes",
        "vector_store_providers",
        "agentic_tools",
        "ingestion_targets",
        "ingestion_upload_extensions",
        "api_key_principals",
        "entra_api_audience",
        "ingestion_default_acl",
        "doc_types",
        "confluence_default_acl",
        "graph_community_acl_scope",
        "rag_selectable_models",
        mode="before",
    )
    @classmethod
    def _split_lists(cls, value: object) -> object:
        return _csv_list(value)

    @field_validator("ingestion_upload_extensions")
    @classmethod
    def _known_upload_extensions(cls, value: list[str]) -> list[str]:
        from app.rag.ingestion.parsing import SUPPORTED_EXTENSIONS

        normalised = [f".{e.strip().lower().lstrip('.')}" for e in value if e.strip()]
        unknown = sorted(set(normalised) - set(SUPPORTED_EXTENSIONS))
        if unknown:
            raise ValueError(f"the parser does not support {', '.join(unknown)}")
        return normalised

    # ----------------------------------------------------------------------------------
    @property
    def utility_deployment(self) -> str:
        return self.azure_openai_utility_deployment or self.azure_openai_chat_deployment

    @property
    def answer_models(self) -> list[str]:
        """Deployments the admin playground may generate answers with (chat first)."""
        configured = self.rag_selectable_models or [
            self.azure_openai_utility_deployment,
            self.azure_openai_fallback_chat_deployment,
        ]
        return list(dict.fromkeys(m for m in [self.azure_openai_chat_deployment, *configured] if m))

    @property
    def active_vector_providers(self) -> list[VectorProvider]:
        """Stores ingestion writes to, default first."""
        return list(dict.fromkeys([self.vector_store_provider, *self.vector_store_providers]))

    def vector_provider_configured(self, provider: str) -> bool:
        return bool(self.mongodb_url.get_secret_value())

    @property
    def vector_store_configured(self) -> bool:
        return all(self.vector_provider_configured(p) for p in self.active_vector_providers)

    @property
    def vector_enabled(self) -> bool:
        """Vector store is needed for the semantic / hybrid modes, or agentic mode with a
        vector tool."""
        vector_tools = {"vector", "keyword", "fetch_document"}
        vector_modes = {RagMode.SEMANTIC, RagMode.HYBRID, RagMode.HYBRID_GRAPH}
        return bool(vector_modes & set(self.rag_enabled_modes)) or (
            RagMode.AGENTIC in self.rag_enabled_modes
            and bool(vector_tools & set(self.agentic_tools))
        )

    @property
    def graph_enabled(self) -> bool:
        """Graph store is needed for the graph / hybrid_graph modes, or for agentic mode with a
        graph tool."""
        graph_tools = {"graph", "graph_global", "graph_query"}
        graph_modes = {RagMode.GRAPH, RagMode.HYBRID_GRAPH}
        return bool(graph_modes & set(self.rag_enabled_modes)) or (
            RagMode.AGENTIC in self.rag_enabled_modes
            and bool(graph_tools & set(self.agentic_tools))
        )

    def configuration_errors(self) -> list[str]:
        """Missing settings for the enabled modes. Logged at startup, failed by readiness."""
        errors: list[str] = []
        if self.rag_mode != RagMode.AUTO and self.rag_mode not in self.rag_enabled_modes:
            errors.append(f"RAG_MODE={self.rag_mode} is not in RAG_ENABLED_MODES")
        if RagMode.AUTO in self.rag_enabled_modes:
            errors.append("RAG_ENABLED_MODES lists real modes only; use RAG_MODE=auto to route")
        if not self.azure_openai_endpoint:
            errors.append("AZURE_OPENAI_ENDPOINT is required")
        mongo = bool(self.mongodb_url.get_secret_value())
        if self.vector_enabled and not self.vector_store_configured:
            errors.append("MONGODB_URL is required for semantic/hybrid/agentic(vector) RAG")
        if self.graph_enabled and not mongo:
            errors.append("MONGODB_URL is required for graph RAG")
        if "entra" in self.auth_mode and not (self.entra_tenant_id and self.entra_api_audience):
            errors.append("ENTRA_TENANT_ID and ENTRA_API_AUDIENCE are required for Entra auth")
        if "api_key" in self.auth_mode and not self.api_key.get_secret_value():
            errors.append("API_KEY is required when AUTH_MODE includes api_key")
        if self.image_parser in ("docintel", "docintel+vision") and not (
            self.doc_intelligence_endpoint
        ):
            errors.append(f"IMAGE_PARSER={self.image_parser} needs DOC_INTELLIGENCE_ENDPOINT")
        if self.image_parser in ("vision", "docintel+vision") and not (
            self.azure_openai_vision_deployment
        ):
            errors.append(f"IMAGE_PARSER={self.image_parser} needs AZURE_OPENAI_VISION_DEPLOYMENT")
        if self.content_safety_mode == "azure" and not self.content_safety_endpoint:
            errors.append("CONTENT_SAFETY_ENDPOINT is required for CONTENT_SAFETY_MODE=azure")
        if self.reranker == "cohere" and not self.reranker_endpoint:
            errors.append("RERANKER_ENDPOINT is required for RERANKER=cohere")
        if self.manifest_store == "mongo" and not mongo:
            errors.append("MONGODB_URL is required for MANIFEST_STORE=mongo")
        if self.zones_store == "blob" and not self.data_lake_account_url:
            errors.append("DATA_LAKE_ACCOUNT_URL is required for ZONES_STORE=blob")
        if self.pii_mode == "azure_language" and not self.language_endpoint:
            errors.append("LANGUAGE_ENDPOINT is required for PII_MODE=azure_language")
        if self.conversation_store == "cosmos" and not self.cosmos_endpoint:
            errors.append("COSMOS_ENDPOINT is required for CONVERSATION_STORE=cosmos")
        return errors

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        vault_url = env_settings().get("azure_key_vault_url") or dotenv_settings().get(
            "azure_key_vault_url"
        )
        if not vault_url:
            return init_settings, env_settings, dotenv_settings, file_secret_settings

        from app.core.azure_auth import get_sync_azure_credential
        from app.core.keyvault import KeyVaultSettingsSource

        key_vault = KeyVaultSettingsSource(
            settings_cls, _url(str(vault_url)), get_sync_azure_credential()
        )
        return init_settings, env_settings, key_vault, dotenv_settings, file_secret_settings


@lru_cache
def get_settings() -> Settings:
    """Settings are cached for the process lifetime; override via env vars per environment."""
    return Settings()

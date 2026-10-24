"""Application settings.

Where each value comes from (highest priority first):
  1. Environment variables — URLs, flags, tuning and secrets.
  2. `.env` file — local development only (never committed).
  3. Defaults below.
Empty values are ignored at every level, so a blank `KEY=` never masks a real value.
"""

import json
import os
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    # --- Authentication & access control (C-048, C-154) ----------------------------------
    # gateway - trust the identity headers the API gateway injects (X-User-ID, X-User-Email,
    #           X-User-Name, X-Account-ID, X-Is-Admin). The gateway authenticates every request
    #           against auth-service and strips any identity headers the caller sent; restrict
    #           this service to the gateway at the network level. Requests without X-User-ID
    #           get 401. Same contract as makemerich-backend and llm-gateway.
    # none    - no authentication; ACLs are NOT enforced (local development only)
    auth_mode: Literal["none", "gateway"] = "none"
    # Ingestion/deletion endpoints need X-Is-Admin (gateway mode). Off = any signed-in user.
    ingestion_requires_admin: bool = False
    # The admin RAG studio (/admin/*: mode comparison, evaluation) needs X-Is-Admin.
    admin_requires_admin: bool = True
    # Enforce document ACLs in every retriever (disable only for single-audience corpora).
    # Principals are the caller's user id and account id (plus "*").
    acl_enforcement: bool = True
    # ACL applied to ingested documents that carry none ("*" = everyone in the org).
    ingestion_default_acl: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    # Requests per minute per user (0 = unlimited). Shared across instances via Redis.
    rate_limit_per_minute: int = 60
    redis_url: SecretStr = SecretStr("")  # redis://:<password>@<host>:6379

    # --- Guardrails (C-156, C-194, C-195) -------------------------------------------------
    content_safety_mode: Literal["off", "heuristic"] = "heuristic"
    block_output_pii: bool = True
    log_pii_masking: bool = True

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

    # --- LLM gateway ----------------------------------------------------------------------
    # Every chat and embedding call goes to the LLM gateway, as the user making the request
    # (their bearer token is forwarded). Which model answers, and what happens when it fails,
    # is the gateway's configuration. See app/rag/generation/llm_gateway.py.
    api_gateway_url: Url = ""  # e.g. http://localhost:8000; LLM calls go to <this>/api/llm
    llm_gateway_url: Url = ""  # direct llm-gateway URL; overrides API_GATEWAY_URL + /api/llm
    # Service key (X-API-Key) for calls made with no caller token: CLI ingestion, scripts,
    # AUTH_MODE=none. Blank = such calls are refused by the gateway.
    llm_gateway_api_key: SecretStr = SecretStr("")
    llm_caller: str = "knowledge-service"  # caller name in the gateway's per-caller stats
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    # Must match the gateway's embedding model: the Atlas vector index is built with it.
    embedding_dimensions: int = 1536  # llm-gateway default: text-embedding-3-small
    embedding_batch_size: int = 64
    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # --- MongoDB Atlas: chunks (vector + keyword search), knowledge graph, manifest --------
    # mongodb+srv://user:pass@cluster/... Vector and keyword search
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
    mongodb_graph_prefix: str = Field(default="rag_graph", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
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
    ingestion_local_root: str = "./data"
    ingestion_max_upload_mb: int = 50
    # File types the upload endpoint / RAG Studio accept (checked by extension and content).
    # Any of the parser's SUPPORTED_EXTENSIONS may be added.
    ingestion_upload_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".xlsx", ".txt", ".json", ".xml",
        ]
    )  # fmt: skip
    # Which stores ingestion writes to by default ("graph" runs LLM entity extraction).
    ingestion_targets: Annotated[list[IngestionTarget], NoDecode] = Field(
        default_factory=lambda: ["vector"]
    )
    ingestion_document_concurrency: int = 4
    ingestion_default_collection: str = "default"

    # --- Connectors (A4) — blank = connector disabled ---------------------------------------
    sources_config_path: str = "config/sources.yaml"  # db:// and api:// definitions
    acl_mapping_path: str = "config/acl_mapping.yaml"
    confluence_base_url: Url = ""  # TODO: https://<org>.atlassian.net/wiki
    confluence_user: str = ""
    confluence_api_token: SecretStr = SecretStr("")
    confluence_default_acl: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    web_crawl_enabled: bool = False
    web_crawl_max_depth: int = 2
    web_crawl_max_pages: int = 200
    web_crawl_delay_seconds: float = 1.0
    web_crawl_auth_header: SecretStr = SecretStr("")  # e.g. "Authorization: Bearer ..."

    # --- Parsing (A5) ------------------------------------------------------------------------
    # PDFs are parsed natively (pypdf): text only. Images and audio/video are not ingested:
    # the LLM gateway takes text-only messages, so there is no vision or speech model.
    pdf_parser: Literal["native"] = "native"

    # --- Cleaning & governance (A6) --------------------------------------------------------------
    ingestion_max_file_mb: int = 100
    ingestion_min_text_chars: int = 50
    ingestion_min_parse_quality: float = 0.4
    ingestion_max_sensitivity: Literal["public", "internal", "confidential", "restricted"] = (
        "confidential"
    )
    ingestion_injection_action: Literal["quarantine", "flag"] = "quarantine"
    pii_mode: Literal["off", "regex"] = "off"
    pii_action: Literal["mask", "flag"] = "mask"
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
    zones_store: Literal["off", "local"] = "local"
    zones_local_root: str = ".ragdata"

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
    reranker_endpoint: Url = ""  # Cohere-compatible rerank endpoint
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

    # --- Conversations (C-157, C-191) ---------------------------------------------------------
    # memory (per process) | mongo (MONGODB_URL / MONGODB_DATABASE)
    conversation_store: Literal["memory", "mongo"] = "memory"
    conversation_max_turns: int = 6
    mongodb_conversations_collection: str = "rag_conversations"

    # --- Answer cache (C-165) -----------------------------------------------------------------
    answer_cache_enabled: bool = False
    answer_cache_similarity: float = Field(default=0.95, ge=0.5, le=1.0)
    answer_cache_ttl_seconds: int = 3600

    # --- Evaluation (admin RAG studio) --------------------------------------------------------
    eval_golden_set_path: str = "evals/golden_set.jsonl"
    eval_thresholds_path: str = "evals/thresholds.yaml"

    # Allow comma-separated lists in env vars.
    @field_validator(
        "cors_allow_origins",
        "rag_enabled_modes",
        "vector_store_providers",
        "agentic_tools",
        "ingestion_targets",
        "ingestion_upload_extensions",
        "ingestion_default_acl",
        "doc_types",
        "confluence_default_acl",
        "graph_community_acl_scope",
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
        if not (self.llm_gateway_url or self.api_gateway_url):
            errors.append("API_GATEWAY_URL (or LLM_GATEWAY_URL) is required for LLM calls")
        mongo = bool(self.mongodb_url.get_secret_value())
        if self.vector_enabled and not self.vector_store_configured:
            errors.append("MONGODB_URL is required for semantic/hybrid/agentic(vector) RAG")
        if self.graph_enabled and not mongo:
            errors.append("MONGODB_URL is required for graph RAG")
        if self.reranker == "cohere" and not self.reranker_endpoint:
            errors.append("RERANKER_ENDPOINT is required for RERANKER=cohere")
        if self.manifest_store == "mongo" and not mongo:
            errors.append("MONGODB_URL is required for MANIFEST_STORE=mongo")
        if self.conversation_store == "mongo" and not mongo:
            errors.append("MONGODB_URL is required for CONVERSATION_STORE=mongo")
        return errors


@lru_cache
def get_settings() -> Settings:
    """Settings are cached for the process lifetime; override via env vars per environment."""
    return Settings()

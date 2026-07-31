"""Knowledge Service — RAG domain configuration.

Env-var prefix: ``RAG_``  (e.g. ``RAG_MONGO_URI``, ``RAG_RETRIEVAL_TOP_K``)

This is deliberately separate from ``app/config.py`` (prefix ``KNOWLEDGE_``):
that file owns *how the service is exposed over HTTP*, this one owns *what the
RAG pipeline does and which LLM Gateway it calls*. Keeping them apart means
this package stays portable — copyable into another service independently of
the HTTP wrapper — exactly like ``features/config.py`` / ``app/config.py`` in
the LLM Gateway this service was extracted from.

Vector database: MongoDB Atlas with ``$vectorSearch`` aggregation — unchanged
from the source implementation (``portless/backend/shared/rag``); no other
vector DB is used.

Embeddings and chat generation: this service holds NO provider API key and
imports NO provider SDK (no OpenAI/Anthropic/Groq/Gemini client anywhere in
this package). Every embedding and LLM call is made over HTTP to an LLM
Gateway instance — see ``RAG_GATEWAY_*`` below and ``rag/llm_gateway_client.py``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent
_HERE = Path(__file__).parent


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        # Load root .env first, then package-local .env (package wins on conflict) —
        # same load order convention as the LLM Gateway's features/config.py.
        env_file=(str(_ROOT / ".env"), str(_HERE / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ──────────────────────────────────────────────────────── MongoDB Atlas
    mongo_uri: str = Field(
        default="", validation_alias=AliasChoices("RAG_MONGO_URI", "MONGO_URI")
    )
    mongo_db_name: str = Field(
        default="knowledge_service",
        validation_alias=AliasChoices("RAG_MONGO_DB_NAME", "MONGO_DB_NAME"),
    )
    chunks_collection: str = "chunks"
    parents_collection: str = "parent_docs"
    # Declared for a future MongoDB-backed job registry (see app/services/ingest_service.py
    # — ingestion job status is currently in-process, matching the source implementation).
    ingestion_jobs_collection: str = "ingestion_jobs"
    mongo_max_pool_size: int = 50
    mongo_min_pool_size: int = 2
    mongo_connect_timeout_ms: int = 5_000
    mongo_server_selection_timeout_ms: int = 5_000
    mongo_socket_timeout_ms: int = 30_000

    # ──────────────────────────────────────────────────── Atlas Vector Search
    # Index must exist before queries run — see ensure_indexes() in
    # vector_store.py for programmatic creation or the Atlas UI instructions.
    vector_index_name: str = "vector_index"
    vector_similarity: str = "cosine"      # "cosine" | "euclidean" | "dotProduct"
    # Must match the dimension of whatever embedding model the configured LLM
    # Gateway actually serves (its own LLM_EMBEDDING_DIMENSIONS) — this
    # service has no in-process access to that setting since the gateway is a
    # separate deployment reached over HTTP. Changing this invalidates every
    # vector already stored.
    embedding_dimensions: int = 1536

    # ──────────────────────────────────────────────────────────── Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120
    semantic_breakpoint_percentile: int = 25
    child_chunks_per_parent: int = 3

    # ──────────────────────────────────────────────────────────── Retrieval
    retrieval_top_k: int = 8
    # numCandidates should be 10-20x top_k for good recall on Atlas
    retrieval_num_candidates: int = 100
    # Max documents fetched for in-memory BM25 keyword search
    keyword_search_fetch_limit: int = 500

    # ──────────────────────────────────────────────────────────── Reranking
    use_cross_encoder_reranker: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ──────────────────────────────────────────────────────────── Generation
    rag_temperature: float = 0.0
    rag_max_tokens: int = 2048

    # ─────────────────────────────────────────────────────── Document loading
    enable_pdf_layout_extraction: bool = True
    enable_pdf_ocr: bool = False

    # ──────────────────────────────────────── Uploaded-file storage (raw bytes)
    # Raw uploaded knowledge files are persisted in MongoDB GridFS — the sole
    # backend (no S3/blob store). GridFS lives in the same shared database
    # (mongo_db_name) as every other collection.
    file_store_gridfs_bucket: str = "rag_files"

    # ──────────────────────────────────────────────────────────── Security
    use_llm_prompt_injection_classifier: bool = False

    # ──────────────────────────────────────────────────────────── Evaluation
    advanced_eval_provider: str = "local"  # "local" | "ragas" | "deepeval" | "llm"

    # ───────────────────────────────────────────── graph retrieval (hybrid RAG)
    # Set graph_retrieval_enabled=true to add a relationship-aware retrieval leg
    # alongside vector + keyword search. Backends: "mongo" (recommended — same
    # shared MongoDB, no new credentials/services), "neo4j" (needs
    # graph_db_uri/user/password), or "arangodb"/"falkordb" (reserved, no-op
    # until wired). When disabled, the query pipeline runs exactly as before
    # (empty graph results, fused harmlessly by RRF).
    graph_retrieval_enabled: bool = False
    graph_retrieval_top_k: int = 8
    graph_db_backend: str = "neo4j"         # "mongo" | "neo4j" | "arangodb" | "falkordb"
    graph_db_uri: str = ""                  # neo4j only, e.g. bolt://localhost:7687
    graph_db_user: str = ""                 # neo4j only
    graph_db_password: str = ""             # neo4j only
    graph_db_database: str = ""             # neo4j only, blank = server default
    graph_chunks_collection: str = "graph_chunks"

    # ──────────────────────────────────── Free web search (hybrid RAG, 4th leg)
    # Adds a web-search retrieval leg (Tavily free tier) alongside vector +
    # keyword + graph search. Disabled by default. The provider credential
    # (TAVILY_API_KEY) is read directly via os.getenv, not RAG_-prefixed — see
    # rag/web_search.py. Absent key or a failed request means no results for
    # that leg (fails open — never fabricates search hits).
    web_search_enabled: bool = False
    web_search_top_k: int = 3

    # ──────────────────────────────────────────────────────── LLM Gateway
    # This service holds no provider credential and imports no provider SDK —
    # every embedding/chat call is an HTTP request to a deployed LLM Gateway
    # instance (see rag/llm_gateway_client.py, rag/llm_gateway_sdk.py — the
    # latter is a vendored copy of that project's sdk/client.py, exactly the
    # way its own docstring says to consume it: "drop this into any
    # application that should reach an LLM through the gateway"). Point this
    # at your LLM Gateway deployment.
    gateway_base_url: str = "http://localhost:8080"
    gateway_api_key: str = ""
    gateway_timeout_seconds: float = 120.0

    # ──────────────────────────────────────────────────────── auth (JWT)
    # Mirrors the LLM Gateway's LLM_AUTH_ENABLED / LLM_JWT_* — an alternative
    # to KNOWLEDGE_API_KEYS (app/config.py) for service-to-service auth on
    # this service's own API.
    auth_enabled: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "knowledge-service"
    jwt_audience: str = "knowledge-service"


settings = RAGSettings()

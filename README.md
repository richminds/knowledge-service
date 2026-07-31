# Knowledge Service

A generic, standalone RAG (Retrieval-Augmented Generation) service: document
ingestion (Markdown / text / PDF, recursive or semantic chunking, parent-child
context) and retrieval (hybrid semantic + keyword + optional graph + optional
web search, reranking, knee-point selection, prompt-injection screening, cited
generation with evaluation metrics) over MongoDB Atlas.

Extracted from Portless's `backend/shared/rag` so any application — Portless
or otherwise — can use it over HTTP instead of embedding the pipeline in its
own codebase. It is a 1:1 feature port: every retrieval leg, every chunking
strategy, every endpoint the source implementation had, this service has too.
Portless's own copy of the code is untouched; this service is additive until
a follow-up migrates Portless to call it instead.

This service holds **no LLM provider API key** and imports **no provider
SDK**. Every embedding and chat-completion call is made over HTTP to a
deployed [`llm-gateway`](../llm-gateway) instance — see
[Why route through the LLM Gateway](#why-route-through-the-llm-gateway) below.

## Quick start

```bash
cp .env.example .env
# Fill in RAG_MONGO_URI (MongoDB Atlas) and RAG_GATEWAY_BASE_URL (a running
# llm-gateway instance) at minimum.

pip install -r requirements-dev.txt
make dev          # http://localhost:8090/docs
```

Ingest a document and ask a question:

```bash
curl -X POST localhost:8090/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"input_paths": ["/data/docs/policy.pdf"], "chunk_strategy": "recursive"}'
# -> {"job_id": "...", "status": "pending", ...}

curl localhost:8090/v1/ingest/<job_id>          # poll until "completed"

curl -X POST localhost:8090/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund window?"}'
```

Or call it from Python with the bundled SDK:

```python
from sdk import KnowledgeServiceClient

client = KnowledgeServiceClient(base_url="http://localhost:8090", api_key="sk-...")
job = await client.ingest(["/data/docs/policy.pdf"])
answer = await client.query("What is the refund window?")
print(answer.answer, answer.sources)
await client.aclose()
```

## Why route through the LLM Gateway

This service does the retrieval and orchestration; a separate
[`llm-gateway`](../llm-gateway) deployment does every model call. That split
means:

- **No provider credentials here.** A leak of this service's config exposes
  nothing that can spend money on a provider account.
- **One place to rotate keys, cap spend, and see traffic** across every
  application that generates text — this service is one more caller of the
  gateway, tracked and rate-limited exactly like any other.
- **Model changes are a gateway-side config change**, not a redeploy of this
  service.

Concretely: `rag/embeddings.py` and `rag/llm.py` never call OpenAI/Anthropic/
Gemini/Groq — they call `rag/llm_gateway_client.py`, which holds one
`RemoteLLMClient` and one `GatewayEmbeddingsClient` (vendored from the
gateway's own `sdk/client.py`, per that module's docstring: "drop this into
any application that should reach an LLM through the gateway"). Point
`RAG_GATEWAY_BASE_URL` / `RAG_GATEWAY_API_KEY` at a deployed instance.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/ingest` | required | Start ingestion from server-side paths |
| `POST` | `/v1/upload` | required | Upload files (multipart), persist to GridFS, then ingest |
| `GET` | `/v1/ingest/{job_id}` | required | Poll ingestion job status |
| `POST` | `/v1/query` | required | Run the full RAG pipeline |
| `DELETE` | `/v1/documents/{source}` | required | Cascading delete (vector + parent + graph) for a source |
| `GET` | `/v1/files` | required | List files in the GridFS file store |
| `GET` | `/v1/stats` | admin | Chunk count + active config snapshot |
| `GET` | `/v1/config` | admin | Resolved, non-secret configuration |
| `GET` | `/health`, `/health/live`, `/health/ready` | none | Liveness/readiness (Mongo + LLM Gateway reachability) |

"required" means any authenticated caller (a configured `KNOWLEDGE_API_KEYS`
entry or a valid JWT) — there is no human-staff-only gate here, unlike the
Portless-specific `require_portless` the source router used, because this
service is meant to be called by any number of applications, not just one
platform's staff UI. "admin" additionally requires `role=admin` on that
credential.

### Who uploaded what

`POST /v1/ingest` and `POST /v1/upload` both accept an optional `user_id` —
the end-user this ingestion is on behalf of (mirrors `QueryRequest.user_id`,
retrieval's equivalent). It's recorded as `uploaded_by` on every resulting
chunk's metadata, every parent-context document, and the GridFS file record
(`GET /v1/files` returns it too) — provenance, not access control. When
omitted, it defaults to the authenticated caller's own principal (the calling
*application's* identity, e.g. `portless-backend`) rather than being left
blank, via `app/dependencies.py::resolve_user_id`.

This is deliberately separate from `authorized_users`/`authorized_teams`
(rag/loader.py), which still default to `"*"` (public) regardless of who
uploaded a document — narrowing visibility to the uploader by default is a
policy decision for later, once there's a concrete requirement for it; today
this only makes authorship queryable/auditable.

## Request flow

```
POST /v1/query
  → RequestContextMiddleware   (request ID, access log)
  → AuthMiddleware             (API key or JWT → principal + role)
  → query_controller → query_service → rag.retrieval.query()
      embed_query → semantic_search → keyword_search → graph_search → web_search
      → hybrid_fusion (RRF) → parent_child_expand → rerank → knee_point_select
      → safety_filter (prompt-injection) → augment → generate (→ LLM Gateway)
      → validate_citations → evaluate
```

Ingestion is fire-and-forget via FastAPI `BackgroundTasks`, tracked in an
**in-process** job registry — status is lost on restart and not shared across
workers, exactly matching the source implementation. Run this service with a
single worker (the Dockerfile default) and scale by replica, not worker
count, until that becomes a MongoDB-backed registry (`RAG_INGESTION_JOBS_COLLECTION`
is declared for this but not yet wired up — a known gap carried over
faithfully from the source, not introduced here).

## Observability

Same pattern as the LLM Gateway this service calls — no OpenTelemetry/
Prometheus/Sentry, just:

- **Structured JSON logs** (`app/logging_config.py`), one object per line,
  `KNOWLEDGE_LOG_FORMAT=text` for a local terminal.
- **Request-ID correlation** (`rag/log_context.py`) — a contextvar bound once
  per request by `RequestContextMiddleware`, read automatically by every log
  line emitted while handling it, including deep inside the pipeline and the
  LLM Gateway client calls, with zero parameter threading.
- **Access log** — one line per request (method, path, status, latency,
  principal), health-check paths excluded to avoid noise.
- **`GET /v1/stats` / `GET /v1/config`** — operational visibility, not
  service traffic, mirroring the gateway's own admin endpoints.
- **`GET /health/ready`** — reports MongoDB and LLM Gateway reachability
  individually, so a caller can tell which dependency degraded.

## Feature status

| Feature | Default | Notes |
|---|---|---|
| Recursive chunking | on | `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` |
| Semantic chunking | opt-in per request | `chunk_strategy: "semantic"` — costs one embedding call per paragraph |
| Parent-child context expansion | always on | broader context returned even though children were matched |
| Hybrid retrieval (semantic + keyword) | always on | Atlas `$vectorSearch` + in-memory BM25 |
| Graph retrieval (4th→3rd leg) | off | `RAG_GRAPH_RETRIEVAL_ENABLED=true`, backend `mongo` (default recommendation) or `neo4j` |
| Web search (4th leg) | off | `RAG_WEB_SEARCH_ENABLED=true` + `TAVILY_API_KEY`, fails open to no results |
| Cross-encoder reranking | off | `RAG_USE_CROSS_ENCODER_RERANKER=true`, else lexical local rerank |
| Prompt-injection screening | heuristic always on | LLM classifier opt-in via `RAG_USE_LLM_PROMPT_INJECTION_CLASSIFIER=true` |
| Citation validation | always on | flags missing/invalid `[n]` markers in the generated answer |
| Answer evaluation | local metrics always on | faithfulness, precision, recall, answer relevance, blended into `confidence_score` (0-1) + `confidence_level` (high/medium/low) on every query response; `RAG_ADVANCED_EVAL_PROVIDER=ragas\|deepeval\|llm` for deeper scoring |
| PDF layout extraction (tables/figures) | on | `pdfplumber`, falls back to `pypdf` |
| PDF OCR | off | `RAG_ENABLE_PDF_OCR=true`, needs system `poppler` + `tesseract` (not in the base image) |
| Authorization filtering | always on | `authorized_users` / `authorized_teams` metadata, `*` = public |

## Layout

```
rag/                          framework-free RAG core (no FastAPI import)
  config.py                     RAGSettings — env prefix RAG_
  auth.py                       JWTValidator (RAG_AUTH_ENABLED)
  log_context.py                contextvar request-ID correlation
  mongo_connection.py           pooled Motor client, index management
  llm_gateway_sdk.py             vendored copy of llm-gateway's sdk/client.py
  llm_gateway_client.py          singleton RemoteLLMClient / GatewayEmbeddingsClient
  embeddings.py                  get_embeddings() → GatewayEmbeddingsClient
  loader.py                      Markdown/text/PDF → LangChain Documents
  chunking.py                    recursive + semantic chunking, parent groups
  vector_store.py                Atlas $vectorSearch, BM25 keyword search, RRF fusion
  parent_store.py                MongoDB parent-document store
  graph_store.py                 Mongo/Neo4j graph-hybrid retrieval, entity extraction
  file_store.py                  MongoDB GridFS raw-file storage
  web_search.py                  Tavily live web-search retrieval leg
  rerank.py                      local lexical + optional cross-encoder reranking
  security.py                    prompt-injection heuristics + optional LLM classifier
  authorization.py               user/team ACL filtering of retrieval results
  citations.py                   validates [n] citation markers
  evaluation.py                  local faithfulness/precision/recall/relevance -> confidence_score + RAGAS/DeepEval/LLM-judge hooks
  llm.py                         prompt construction + generation via the LLM Gateway
  ingestion.py                   LangGraph orchestration — ingestion pipeline only
  retrieval.py                   LangGraph orchestration — query/retrieval pipeline only
app/                           thin FastAPI HTTP layer
  main.py                        create_app(), lifespan, middleware/router wiring
  config.py                      KnowledgeServiceSettings — env prefix KNOWLEDGE_
  dependencies.py                principal/role Depends() providers
  errors.py                      GatewayError → HTTP status mapping
  logging_config.py              JSON/text structured logging
  controllers/                   one router per resource
  models/                        pydantic request/response schemas
  services/                      orchestration between HTTP and rag/
  middleware/                    auth.py, request_context.py
sdk/                           HTTP client for other applications to call this service
scripts/
  mint_token.py                  mint a JWT for local testing
  rag_cli.py                     local ingest/ask CLI, no HTTP server needed
tests/
  conftest.py                    shared fixtures (hermetic env, TestClient)
  unit/                          pure logic, no network/Mongo/FastAPI app
  integration/                   real app/TestClient or real pipeline wiring,
                                  with only the Mongo/LLM Gateway I/O boundary mocked
```

Ingestion and retrieval are deliberately separate modules (`rag/ingestion.py`,
`rag/retrieval.py`) rather than one `graph.py` — they are independent
concerns with different scaling and failure characteristics: ingestion is a
write-heavy background job (see `app/services/ingest_service.py`), retrieval
is a read-heavy, latency-sensitive request path. Either can move to its own
worker/process later without the other's dependencies coming along.

## Configuration

See [`.env.example`](.env.example) for the full, annotated list. Two
prefixes: `KNOWLEDGE_*` (how this service is exposed) and `RAG_*` (what the
pipeline does and which LLM Gateway it calls).

## Deployment checklist

- [ ] `RAG_MONGO_URI` set to an Atlas cluster (Atlas Vector Search requires
      Atlas, or an Atlas-compatible deployment with Search support)
- [ ] `RAG_GATEWAY_BASE_URL` / `RAG_GATEWAY_API_KEY` point at a reachable,
      already-deployed `llm-gateway` instance
- [ ] `RAG_EMBEDDING_DIMENSIONS` matches that gateway's actual
      `LLM_EMBEDDING_DIMENSIONS` (mismatched dimensions corrupt the vector index)
- [ ] `KNOWLEDGE_API_KEYS` or `RAG_AUTH_ENABLED` configured — otherwise the
      service is open to anyone who finds the URL
- [ ] `KNOWLEDGE_DOCS_ENABLED=false` if the API is reachable from the public internet
- [ ] Single worker / replica-based scaling until the ingestion job registry
      is MongoDB-backed (see Request flow above)

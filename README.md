# OISSA Knowledge Agent

A question-answering service over internal documents. It is built with FastAPI and MongoDB Atlas (vector search, keyword search, the knowledge graph and conversations). Every LLM and embedding call goes through the LLM gateway, and identity comes from the API gateway, the same way as makemerich-backend.

It offers five retrieval strategies, which can be switched with a flag and validated side by side, plus a router that picks between them:

| Mode | Best for | How it retrieves |
|---|---|---|
| `semantic` | Paraphrased, meaning-based questions | Dense vector search only |
| `hybrid` | Factual "what / how" questions, codes and names | Vector + keyword (BM25) search, fused with Reciprocal Rank Fusion |
| `hybrid_graph` | Factual questions that also involve relationships | `hybrid` search and the knowledge graph run in parallel; chunks are fused with RRF and the graph facts are added in front |
| `graph` | Relationships, multi-hop and "big picture" questions | A knowledge graph (entities, relations, communities) in MongoDB collections |
| `agentic` | Multi-step and comparative questions | A LangGraph agent that plans, calls search tools, grades evidence and checks its answer |
| `auto` | Production traffic | Routes each question to the cheapest suitable mode, and escalates when confidence is low |

Every answer has numbered citations, a confidence label (high / medium / low) with a plain-language reason, and token usage. Access control is enforced in every retriever: users only get answers from documents they are allowed to see.

> **New here?** Read [docs/workflow-execution.md](docs/workflow-execution.md). It walks through how to run everything, step by step, from a laptop to production.

---

## Contents

- [Architecture](#architecture)
- [Quick start (local)](#quick-start-local)
- [Configuration and secrets](#configuration-and-secrets)
- [Feature flags](#feature-flags)
- [Choosing the RAG mode](#choosing-the-rag-mode)
- [API](#api)
- [Answer quality](#answer-quality)
- [Security](#security)
- [Observability](#observability)
- [Testing and evaluation](#testing-and-evaluation)
- [Deployment](#deployment)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Known gaps](#known-gaps)

---

## Architecture

```text
        Sources (local files, uploads, Confluence, DB, REST, web)
                                  │
   detect type ─► parse ─► clean & govern ─► metadata ─► chunk ─► enrich ─► embed
                                  │                                    │
                  MongoDB (graph collections)         MongoDB Atlas Search (vectors + BM25)
                                  │                                    │
 User ─► API gateway (auth-service sign-in, injects X-User-*) ─► rate limit ─► question analysis
        ─► router (semantic | hybrid | hybrid_graph | graph | agentic)
        ─► fuse / rerank / context budget ─► LLM gateway (via the API gateway, as the user)
        ─► citations + confidence ─► streamed answer (SSE)
```

| Concern | Service |
|---|---|
| Identity | API gateway: authenticates against auth-service, strips caller-sent identity headers, injects `X-User-ID`, `X-User-Email`, `X-User-Name`, `X-Account-ID`, `X-Is-Admin` |
| Models (chat and embeddings) | LLM gateway, reached at `API_GATEWAY_URL` + `/api/llm` with the caller's bearer token. It picks the model, retries and fails over; this service holds no provider key. |
| Search, graph, manifest, conversations | MongoDB Atlas (Vector Search + Atlas Search), `MONGODB_URL` |
| Optional | Redis (shared rate limits and caches), a Cohere-compatible reranker |

More detail is in [docs/architecture.md](docs/architecture.md) and the decision records in [docs/adr/](docs/adr/).

---

## Quick start (local)

**Prerequisites:**
- Python 3.11+ (3.12 recommended).
- A running LLM gateway (llm-gateway), either behind the API gateway or directly, and a key for it if you call it without a signed-in user.
- A MongoDB Atlas cluster (the free M0 tier works) whose network access list allows your IP. Networks that decrypt TLS can block Atlas authentication.

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
make install                              # pip install -r requirements-dev.txt + pre-commit

# 2. Configure: copy the template, then set the gateway and the Atlas connection string
copy .env.example .env                    # macOS/Linux: cp .env.example .env
#    API_GATEWAY_URL=http://localhost:<api-gateway-port>   (or LLM_GATEWAY_URL=http://localhost:8080)
#    LLM_GATEWAY_API_KEY=sk-...            (needed while AUTH_MODE=none: no user token to forward)
#    MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/

# 3. Start the API (collections and Atlas Search indexes are created on startup)
make dev                                  # http://localhost:8090 (api-gateway routes /api/knowledge here)

# 4. Load the sample documents and ask a question
make ingest
```

Then open **http://localhost:8090/ui** for the chat page, or **http://localhost:8090/docs** for the interactive API:

```bash
curl -X POST localhost:8090/api/v1/chat -H "Content-Type: application/json" \
  -d '{"query": "Which file formats are supported?"}'
```

Locally, `AUTH_MODE=none`, so no sign-in is needed and ACLs are not enforced. With no signed-in user there is no token to forward, so LLM calls use `LLM_GATEWAY_API_KEY` instead. Set `EMBEDDING_DIMENSIONS` to the size of the gateway's embedding model before the first ingest: the vector index is created with it.

---

## Configuration and secrets

Every setting is an environment variable. [.env.example](.env.example) lists all of them, with comments and `TODO` placeholders.

| What | Where |
|---|---|
| Service URLs, flags, tuning and secrets (`MONGODB_URL`, `LLM_GATEWAY_API_KEY`, connector credentials) | Environment variables, or `.env` locally |
| Source lists, ACL mapping, glossary, ontology, Cypher templates | `config/*.yaml`, baked into the image |

- **Precedence** (highest first): environment variables → `.env` → defaults. A blank value never overrides anything.
- **Validation:** missing or invalid settings for the enabled features are logged at startup and reported by `GET /api/v1/health/ready`. The app does not crash-loop.

`.env` is git-ignored. Never commit secrets.

---

## Feature flags

Most steps of the question and ingestion pipelines can be switched on or off. [docs/feature-flags.md](docs/feature-flags.md) shows each step, the flag that controls it, what changes when it's on or off, which flags add LLM calls, and ready-made profiles (cheapest, best quality, evaluation, production).

The flags you're most likely to change:

| Flag | Default | What it switches |
|---|---|---|
| `RAG_MODE` / `RAG_ENABLED_MODES` | `semantic` / `semantic` | Default retrieval strategy, and which strategies are available |
| `QUERY_ANALYSIS_ENABLED` | `true` | LLM question analysis: follow-up rewrite, intent, filters |
| `ASSISTANT_SCOPE` | blank | What counts as in scope. Blank or too narrow, and valid questions get the "outside what I can help with" reply. |
| `RETRIEVAL_MODE` | `hybrid` | `vector`, `hybrid` (vector + BM25) or `semantic_hybrid` (+ semantic reranker) |
| `SEMANTIC_QUERY_TRANSFORM` | `none` | `multi_query`, `hyde` or `step_back` query rewriting |
| `RERANKER` | `none` | `llm` or `cohere` reranking |
| `RETRIEVAL_KNEE_CUTOFF` | `false` | Drops passages after the sharp drop in scores |
| `QUALITY_JUDGE` | `request` | LLM judge (groundedness, relevance): `off`, `request` or `always` |
| `ANSWER_CACHE_ENABLED` | `false` | Reuses answers to similar questions. Keep it off while testing. |
| `RAG_ESCALATION_ENABLED` | `false` | Retries low-confidence answers in a stronger mode |
| `INGESTION_TARGETS` | `vector` | Adds the knowledge graph with `graph` |
| `CHUNK_PARENT_SIZE`, `ENRICH_*` | off | Parent-child chunking and LLM enrichment at ingestion (re-ingest with `--force` after changing) |

Restart the API after changing any flag.

---

## Choosing the RAG mode

| Setting | Purpose | QA | Prod |
|---|---|---|---|
| `RAG_MODE` | Default mode: `semantic`, `hybrid`, `hybrid_graph`, `graph`, `agentic` or `auto` | `semantic` | `hybrid` (change after validation) |
| `RAG_ENABLED_MODES` | Modes this environment builds and accepts | all three | only the validated ones |
| `RAG_ALLOW_MODE_OVERRIDE` | Lets a request pick `"rag_mode"` | `true` | `false` |
| `RAG_ESCALATION_ENABLED` | Retries a low-confidence answer with a stronger mode | `true` | `false` |
| `INGESTION_TARGETS` | Where ingestion writes: `vector`, `graph` | `vector,graph` | `vector` |

To compare the modes on the same question, add `"rag_mode"` and `"include_trace": true` to a chat request. To compare them on the whole golden set, run `python scripts/run_eval.py --rag-mode all --judge`.

How each mode works:
- **Semantic:** question analysis (follow-up rewrite, intent, filters) → optional multi-query or HyDE → hybrid search with ACL filters → MMR → parent or sentence-window expansion → rerank.
- **Graph:** entity linking → local, path, hybrid, global (community reports), DRIFT, dual-level or Cypher-template search → PageRank ranking. It falls back to semantic search when no entities match.
- **Agentic:** route → plan → retrieve with tools (`AGENTIC_TOOLS`) → grade → refine → generate → check. Steps, tokens and time are limited.

### Vector and keyword search on MongoDB Atlas

Chunks are documents in `MONGODB_CHUNKS_COLLECTION` (`_id` = chunk id). Two Atlas Search indexes are created on startup (or with `make create-index`). They build in the background, so a new collection answers searches a few seconds after the first write.

| Step | How |
|---|---|
| Vector | `$vectorSearch` (HNSW, cosine) on `MONGODB_VECTOR_INDEX`; `numCandidates` = top_k × `MONGODB_NUM_CANDIDATES_FACTOR` |
| Keyword | `$search` (Lucene BM25, `MONGODB_TEXT_ANALYZER`) over title, section path and content on `MONGODB_SEARCH_INDEX` |
| Filters | Collection, ACL, kind and metadata filters are applied inside the query (a `$vectorSearch` pre-filter, or `$search` compound filters) |
| Hybrid fusion | RRF in the app, k = 60 |
| Semantic reranking (`semantic_hybrid`) | `MONGODB_SEMANTIC_RERANKER`: `llm` (slower), `cohere` or `none` |

The graph lives in the same database, in `<MONGODB_GRAPH_PREFIX>_*` collections: chunks with their mentioned entity keys, entities (with description embeddings for entity linking), relations, claims, sections and documents. Traversal runs hop by hop with indexed queries. There is no Cypher engine, so `GRAPH_SEARCH_MODE=cypher` and the agent's `graph_query` tool fall back to local graph search.

`VECTOR_STORE_PROVIDERS`, the per-request `"vector_store"` field and `--vector-store all` still exist so another store can be added for side-by-side comparison. Today the only provider is `mongo`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health`, `/api/v1/health/ready` | Liveness and readiness (config + every enabled store) |
| GET | `/api/v1/rag/modes` | Default mode, enabled modes, and whether override is allowed |
| POST | `/api/v1/chat` | Grounded answer with citations, confidence and usage |
| POST | `/api/v1/chat/stream` | The same, as Server-Sent Events: `step`* → `citations` → `token`* → `quality` → `done` |
| POST | `/api/v1/retrieval` | Retrieved passages only, without an answer |
| POST | `/api/v1/feedback` | 👍 / 👎 and a comment on an answer, by `response_id` |
| DELETE | `/api/v1/conversations/me` | Delete the caller's chat history |
| POST | `/api/v1/ingestion` | Ingestion job for a source URI: a local path, `confluence://`, `db://`, `api://` or `https://` |
| POST | `/api/v1/ingestion/upload` | Ingest one uploaded file (with optional `acl` and `collection`) |
| GET | `/api/v1/ingestion/{job_id}` | Job status |
| GET | `/api/v1/ingestion/manifest` | Status of each document: indexed, skipped, quarantined or failed |
| DELETE | `/api/v1/ingestion/sources?source=` | Remove a document from every store |
| DELETE | `/api/v1/ingestion/collections/{name}` | Remove a whole collection |
| GET | `/ui` | Built-in chat page (`UI_ENABLED`) |
| GET | `/api/v1/admin/rag/config` | Admin: enabled modes, models in play (all `llm-gateway`), retrieval settings, eval thresholds and golden set |
| POST | `/api/v1/admin/rag/evaluate` | Admin: one scored run for a mode (uncached): answer, top-k citations, trace, judge scores and retrieval metrics (hit, recall, MRR, nDCG, citation precision/recall, correctness) |

With `AUTH_MODE=gateway`, everything except the health endpoints and `/ui` requires the `X-User-ID` header the API gateway injects, so call the service through the gateway with an auth-service token. The `/admin` endpoints also require `X-Is-Admin: true` (`ADMIN_REQUIRES_ADMIN`, on by default); they back the RAG Studio page.

---

## Answer quality

Each `/chat` response (and the `quality` event of the stream) includes:

```json
{
  "answer": "Staff receive 25 days of annual leave [1].",
  "citations": [{"index": 1, "title": "Leave policy", "page": 3, "cited": true, "snippet": "..."}],
  "quality": {
    "confidence": "high",
    "explanation": "Every statement cites a source; reviewer scores: supported 5, relevant 4.",
    "signals": {"citation_coverage": 1.0, "invalid_citations": []},
    "judge": {"groundedness": 5, "answer_relevance": 4, "context_relevance": 4}
  },
  "usage": {"total_tokens": 2492, "llm_calls": 2},
  "latency_ms": 2140.5
}
```

- **Signals** are free and computed for every answer: was it answered, citation coverage, and citations that don't exist.
- The **LLM judge** scores groundedness and relevance from 1 to 5. It runs when the request sets `"evaluate": true`, or on a background sample of answers (`QUALITY_JUDGE_SAMPLE_RATE`).
- **Confidence** is **low** when nothing is cited or the judge scores ≤ 2. It is **high** when the answer is well cited and the judge scores ≥ 4. Otherwise it is **medium**.

---

## Security

| Control | Implementation |
|---|---|
| Sign-in | `AUTH_MODE=gateway`: identity headers injected by the API gateway, which validates the auth-service token and strips any identity headers the caller sent. Restrict this service to the gateway at the network level: a direct request could set the headers itself. `none` is for local use only. |
| Document permissions | Every chunk and graph entity carries an ACL. The user's ID and account ID filter **every** store query. The LLM never decides access. |
| Ingestion rights | `INGESTION_REQUIRES_ADMIN` (admins only, from `X-Is-Admin`) |
| Prompt injection | Checked on questions, quarantined at ingestion, and flagged in retrieved passages (pattern checks). |
| Abuse | Per-user rate limits (`RATE_LIMIT_PER_MINUTE`) and input validation |
| Privacy | PII masked in logs, with optional PII masking in documents. Deletion endpoints exist for users and sources. Every query and ingest is audited. |
| LLM spend | Calls go out as the user, so the gateway's per-user rate limits, budgets and metering apply |

`tests/integration/test_security.py` and `test_isolation_redteam.py` prove these controls. The manual checklist is [docs/runbooks/red-team.md](docs/runbooks/red-team.md).

---

## Observability

- **Logs:**
  - JSON to stdout.
  - One `rag_answer` event per answer, with mode, confidence, latency and tokens.
  - The question text is logged only if `LOG_QUESTION_TEXT=true`.
- **Traces and metrics:** OpenTelemetry API calls for every RAG step (`rag.vector_search`, `rag.graph_traversal`, `rag.grade`, …) and `rag.request.duration`, `rag.answers`, `rag.quality.score`, `rag.llm.tokens`, `rag.feedback` plus the ingestion counters. No exporter is configured, so they are no-ops until you add one.
- **LLM usage:** per-caller stats live in the LLM gateway; calls are labelled `LLM_CALLER:<purpose>` (`knowledge-service:answer`, `knowledge-service:query_analysis`, ...).
- **Correlation:** the `X-Request-ID` header equals `response_id`.

---

## Testing and evaluation

```bash
make test                     # unit + integration tests (in-memory fakes; never call the gateway)
make lint                     # ruff lint + format check
make typecheck                # mypy
make eval                     # golden-set retrieval metrics for every enabled mode
python scripts/run_eval.py --rag-mode all --judge --gate     # quality gate against thresholds.yaml
python scripts/run_ablation.py                               # compare configuration variants
locust -f tests/load/locustfile.py --host https://<qa-app>   # load test (QA only)
```

| Eval asset | Purpose |
|---|---|
| `evals/golden_set.jsonl` | Questions with expected sources and answers, and a `question_type` |
| `evals/thresholds.yaml` | Minimum scores the CI gate enforces |
| `evals/results/baseline.json` | The reference run that regressions are measured against (`--save-baseline`) |
| `evals/ablations.yaml` | Variants: chunk size, retrieval mode, reranker, enrichment, graph settings |
| `evals/*_gold.jsonl`, `human_labels.jsonl` | Graph extraction, linking and Cypher accuracy; judge calibration |
| `data/sample/toy-room-inventory.pdf` | A small test PDF for the `toy-q*` golden-set questions. Rebuild it with `python scripts/make_sample_pdf.py`. |

To measure the effect of a flag change, see [docs/feature-flags.md, section 6](docs/feature-flags.md#6-testing-a-flag-change).

---

## Deployment

The service is one container (see the [Dockerfile](Dockerfile)); run it anywhere that can reach MongoDB Atlas and the API gateway, and put it behind the API gateway (its route registry expects knowledge-service under `/api/knowledge`). Configure it with environment variables as in [.env.example](.env.example), with at least `AUTH_MODE=gateway`, `API_GATEWAY_URL` and `MONGODB_URL`. To roll back, redeploy the previous image; see [docs/runbooks/rollback.md](docs/runbooks/rollback.md).

---

## Repository layout

```text
src/app/
  main.py               app factory, lifespan, /ui
  core/                 config, gateway identity (auth), caller token, rate limit, audit, PII, metrics
  api/                  dependency wiring (deps.py) and v1 endpoints
  rag/
    strategies/         semantic (+ hybrid), hybrid_graph, graph, agentic, registry/router
    agents/             agent tools
    query/              question analysis, glossary, query transforms
    retrieval/          retriever, rerankers
    context/            fusion, MMR, ordering, token budget
    vectorstore/        MongoDB Atlas vector + keyword search
    graph/              MongoDB graph store, extraction, ontology, communities
    ingestion/          connectors, parsing, governance, chunking, enrichment, manifest, pipeline
    generation/         prompts and the LLM gateway client (vendored SDK)
    evaluation/         answer quality and the offline harness
  services/             chat/retrieval, ingestion, conversations
  prompts/              versioned prompt templates (YAML)
  static/chat.html      chat UI
config/                 YAML placeholders (sources, ACLs, glossary, ontology, ...)
scripts/                ingest, reconcile, graph maintenance, eval, tooling
evals/                  golden set, thresholds, ablations, gold sets
tests/                  unit/, integration/, load/
docs/                   architecture, ADRs, workflow, runbooks, user guide
```

---

## Documentation

| Document | Read it when |
|---|---|
| [docs/workflow-execution.md](docs/workflow-execution.md) | You want to run, deploy or operate the system |
| [docs/feature-flags.md](docs/feature-flags.md) | You want to know what a flag does, or which flags to turn on or off |
| [docs/architecture.md](docs/architecture.md) | You need the component and security overview |
| [docs/requirements.md](docs/requirements.md) | You are filling in the business inputs (targets, budgets, ACL model) |
| [docs/adr/](docs/adr/) | You want to know why a technology was chosen |
| [docs/user-guide.md](docs/user-guide.md) | You are writing material for end users |
| [docs/runbooks/](docs/runbooks/) | You are re-indexing, rolling back, handling an incident, or running a red team |
| [CONTRIBUTING.md](CONTRIBUTING.md) | You are adding a source, file format, tool or prompt |
| [RAG-E2E-Tasks-Semantic-Graph-Agentic.md](RAG-E2E-Tasks-Semantic-Graph-Agentic.md) | You want the build checklist and its status |

---

## Known gaps

- **Network restriction:** nothing in-process stops a direct caller from sending `X-User-ID`; only allow traffic from the API gateway.
- **Images and audio/video:** not ingested. The LLM gateway takes text-only messages, so there is no OCR, vision or speech model; scanned PDF pages are flagged `pages_without_text`.
- **Per-request model choice:** the gateway picks the model, so the admin studio can no longer compare answer models.
- **MongoDB least privilege:** use an Atlas database user scoped to `MONGODB_DATABASE` (readWrite + search index management), not an admin user.
- **Agents:** a SQL tool, sandboxed code execution and MCP exposure for agent tools.
- **Placeholders to fill:** search the repository for `TODO`. The full list is in [docs/workflow-execution.md](docs/workflow-execution.md#4-fill-in-the-placeholders).

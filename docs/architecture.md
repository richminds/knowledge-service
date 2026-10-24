# Architecture (C-013)

```text
                     ┌──────────────────────── Ingestion ─────────────────────────┐
 local files,        │ connectors ─► detect type ─► parse ─► clean/govern ─► metadata│
 Confluence, web,  ──►  ─► chunk ─► enrich ─► embed ─► MongoDB Atlas (vector+BM25)    │
 DB, REST, upload    │                    └─► extract graph ─► MongoDB graph colls.  │
                     │ zones: raw/parsed/clean/chunks/quarantine (local), manifest (Mongo)│
                     └─────────▲───────────────────────────────▲─────────────────────┘
          CLI / API job / upload                               │ scheduled reconcile + graph scripts
                                                               │
 User ─► API gateway ─► /ui or API ─► auth (gateway headers) ─► rate limit ─► query understanding
        ─► router (RAG_MODE / auto) ─┬─ semantic: hybrid search, MMR, parent/window expansion
                                     ├─ graph:    local, path, global, DRIFT, Cypher templates
                                     └─ agentic:  LangGraph plan → tools → grade → check
        ─► fuse/rerank/context budget ─► LLM gateway ─► citations + confidence ─► SSE
        ─► conversation store (Mongo) · answer cache (Redis) · JSON logs (stdout)
```

## Components

| Layer | Code | External dependency |
|---|---|---|
| API and UI | `src/app/main.py`, `api/v1/`, `static/chat.html` | Container image (`Dockerfile`), reached through the API gateway |
| Auth | `core/auth.py`, `core/caller_context.py`, `core/ratelimit.py` | API gateway (identity headers), Redis (optional) |
| Query understanding | `rag/query/` | LLM gateway |
| Semantic RAG | `rag/strategies/semantic.py`, `rag/retrieval/`, `rag/vectorstore/` | MongoDB Atlas Vector Search + Atlas Search |
| Graph RAG | `rag/strategies/graph.py`, `rag/graph/` | MongoDB collections (entities, relations, claims) + Atlas Vector Search for entity linking |
| Agentic RAG | `rag/strategies/agentic.py`, `rag/agents/tools.py` | LangGraph in-process; MongoDB checkpointer optional |
| Router | `rag/strategies/registry.py`, `services/retrieval_service.py` | none |
| Generation | `rag/generation/`, `prompts/*.yaml` | LLM gateway (`POST /v1/llm`, `POST /v1/embeddings`); it picks the model and fails over |
| Safety | `rag/safety.py`, `core/pii.py` | none (heuristic and regex checks) |
| Ingestion | `rag/ingestion/`, `scripts/ingest_cli.py` | MongoDB; Confluence, web, `db://` and `api://` sources when configured |
| Evaluation | `rag/evaluation/`, `scripts/run_eval.py`, `evals/` | `run_eval.py --gate` |
| Observability | `core/logging.py`, `core/metrics.py` | JSON logs on stdout; OpenTelemetry metrics are recorded but no exporter is configured |

## Security model

- **Identity.** The API gateway authenticates every request against auth-service, strips any identity headers the caller sent, and injects `X-User-ID`, `X-User-Email`, `X-User-Name`, `X-Account-ID` and `X-Is-Admin` (`AUTH_MODE=gateway`). This service validates no tokens, so it must only be reachable through the gateway.
- **Outbound LLM calls.** The caller's bearer token is forwarded to the LLM gateway, so per-user budgets and metering apply there. Calls with no caller (CLI, scripts) use `LLM_GATEWAY_API_KEY`. The service holds no model-provider credential.
- **Secrets.** `MONGODB_URL`, `LLM_GATEWAY_API_KEY`, `REDIS_URL` and the connector credentials are environment variables.
- **Users.** The user ID and account ID, plus `"*"`, become ACL principals, which are enforced inside every store query.
- **Prompt-injection defence.** There are four layers: documents are scanned and quarantined at ingestion; questions are checked; retrieved passages are flagged and delimited as data; answers are checked.

## Not yet provisioned (TODO)

- Network restriction so that only the API gateway can reach the service (required by `AUTH_MODE=gateway`), and egress control (C-049).
- A WAF in front of the API gateway (C-050).
- Log and metric export, and LLM tracing such as Langfuse (C-052).

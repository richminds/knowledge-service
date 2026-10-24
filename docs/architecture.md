# Architecture (C-013)

```text
                     ┌──────────────────────── Ingestion ─────────────────────────┐
 SharePoint, blob,   │ connectors ─► detect type ─► parse ─► clean/govern ─► metadata│
 Confluence, web,  ──►  ─► chunk ─► enrich ─► embed ─► MongoDB Atlas (vector+BM25)    │
 DB, REST            │                    └─► extract graph ─► MongoDB graph colls.  │
                     │ zones: raw/parsed/clean/chunks/quarantine (Blob), manifest (Mongo)│
                     └─────────▲───────────────────────────────▲─────────────────────┘
          Event Grid ─► Service Bus ─► Container Apps job      │ nightly reconcile + graph jobs
                                                              │
 User ─► /ui or API ─► auth (Entra / API key) ─► rate limit ─► query understanding
        ─► router (RAG_MODE / auto) ─┬─ semantic: hybrid search, MMR, parent/window expansion
                                     ├─ graph:    local, path, global, DRIFT, Cypher templates
                                     └─ agentic:  LangGraph plan → tools → grade → check
        ─► fuse/rerank/context budget ─► Azure OpenAI ─► citations + confidence ─► SSE
        ─► conversation store (Cosmos) · answer cache (Redis) · telemetry (App Insights)
```

## Components

| Layer | Code | Azure resource (`infra/`) |
|---|---|---|
| API and UI | `src/app/main.py`, `api/v1/`, `static/chat.html` | App Service (Linux container) + prod staging slot |
| Auth | `core/auth.py`, `core/ratelimit.py` | Entra app registration (manual), Redis (optional) |
| Query understanding | `rag/query/` | Azure OpenAI utility deployment |
| Semantic RAG | `rag/strategies/semantic.py`, `rag/retrieval/`, `rag/vectorstore/` | MongoDB Atlas Vector Search + Atlas Search |
| Graph RAG | `rag/strategies/graph.py`, `rag/graph/` | MongoDB collections (entities, relations, claims) + Atlas Vector Search for entity linking |
| Agentic RAG | `rag/strategies/agentic.py`, `rag/agents/tools.py` | LangGraph in-process; MongoDB checkpointer optional |
| Router | `rag/strategies/registry.py`, `services/retrieval_service.py` | none |
| Generation | `rag/generation/`, `prompts/*.yaml` | Azure OpenAI chat deployment (+ fallback) |
| Safety | `rag/safety.py`, `core/pii.py` | Content Safety, Language (optional) |
| Ingestion | `rag/ingestion/`, `scripts/ingestion_worker.py` | Storage, Event Grid, Service Bus, Container Apps jobs |
| Evaluation | `rag/evaluation/`, `scripts/run_eval.py`, `evals/` | QA environment + pipeline quality gate |
| Observability | `core/telemetry.py`, `core/metrics.py` | Log Analytics, App Insights, alerts, workbook, budget |

## Security model

- **Identity.** One user-assigned identity is shared by the web app and the jobs. It has least-privilege data roles on Storage, Key Vault, OpenAI, Service Bus and Cosmos. No service keys are stored, with two exceptions kept in Key Vault: the Redis URL and the MongoDB Atlas connection string (`mongodb-url`).
- **Users.** Entra tokens are validated against the tenant JWKS. Group IDs and the user ID become ACL principals, which are enforced inside every store query.
- **Prompt-injection defence.** There are four layers: documents are scanned and quarantined at ingestion; questions are checked; retrieved passages are flagged and delimited as data; answers are checked.

## Not yet provisioned (TODO)

- Private endpoints, VNet integration and egress control (C-049).
- API Management or Front Door with WAF in front of the App Service (C-050), and as an LLM gateway (C-038).
- LLM tracing beyond App Insights, such as Langfuse (C-052).

# 0008. LLM gateway for every model call, API-gateway identity, no Azure dependencies (C-038, C-048, C-154)

Supersedes [0005](0005-keyless-identity.md) and [0006](0006-hosting.md), and the Key Vault and Bicep consequences of [0007](0007-mongodb-atlas-store.md).

**Context.** The service called Azure OpenAI directly with its own deployments and a fallback endpoint, validated Entra ID tokens or an API key itself, and depended on a dozen other Azure services: Key Vault, Application Insights, Content Safety, AI Language, Document Intelligence, Blob Storage, SharePoint via Entra, Service Bus, Event Grid, Cosmos DB, App Service and Container Apps jobs, all provisioned by Bicep and deployed by Azure DevOps. The sibling services already share two pieces of platform: the API gateway, which authenticates every request against auth-service and injects the caller's identity, and llm-gateway, which owns model choice, failover, provider keys, per-user budgets and metering. makemerich-backend uses both.

**Decision.**
- Every chat and embedding call goes to llm-gateway (`POST /v1/llm`, `POST /v1/embeddings`), at `API_GATEWAY_URL` + `/api/llm`, or `LLM_GATEWAY_URL` directly. This is the same design as makemerich-backend.
  - The call carries the calling user's bearer token. `CallerTokenMiddleware` binds it per request (`core/caller_context.py`).
  - With no caller (CLI ingestion, scripts, `AUTH_MODE=none`), `LLM_GATEWAY_API_KEY` is sent as `X-API-Key`. With neither, the gateway refuses the call.
  - The gateway picks the model. Each call names only a category (`factual_qa`, `conversational`) and a caller label (`LLM_CALLER:<purpose>`).
- `AUTH_MODE=gateway` trusts the identity headers the API gateway injects: `X-User-ID` (required, else 401), `X-User-Email`, `X-User-Name`, `X-Account-ID` and `X-Is-Admin`. The service validates no tokens. `AUTH_MODE=none` is for local development only.
- ACL principals are `"*"`, the user ID and the account ID. Admin checks use `X-Is-Admin`, behind `ADMIN_REQUIRES_ADMIN` and `INGESTION_REQUIRES_ADMIN`.
- MongoDB (`MONGODB_URL`) holds everything stateful: vectors, keyword search, the graph, the manifest, and now conversations (`CONVERSATION_STORE=mongo`).
- Every other Azure dependency is removed. Settings come from environment variables and `.env` only. Logs are structured JSON on stdout. Guardrails and PII detection are local (heuristic and regex). PDFs are parsed as text with pypdf. Raw-copy zones are `local` or `off`. `infra/` and `pipelines/` are deleted.

**Consequences.**
- No provider credential lives in this service. Rate limits, budgets and token metering per user happen in one place, the gateway.
- **Model choice per request is gone.** `RAG_SELECTABLE_MODELS`, the deployment settings and the fallback endpoint no longer exist. The admin studio offers only `llm-gateway` as the answer model. Comparing models means changing the gateway's configuration, not this service's.
- `EMBEDDING_DIMENSIONS` must match the gateway's embedding model. If the gateway changes that model, the vector index needs a [full re-index](../runbooks/reindex.md).
- Model replies are less uniform, because the gateway may route to different providers. JSON replies are parsed tolerantly (`extract_json_object`).
- **Image and audio/video ingestion is gone.** The gateway takes text-only messages, so there is no OCR, vision or speech model. Image files are no longer accepted, pictures inside PDFs are ignored, scanned PDF pages give no text, and PDFs lose Document Intelligence's layout and table extraction.
- Content Safety's harm-category checks are gone. Only the heuristic injection and PII checks remain. PII detection at ingestion is regex only, so it misses names and addresses.
- Blob and SharePoint sources, event-driven ingestion (Event Grid, Service Bus, the ingestion worker) and the Blob data-lake zones are gone. Content arrives through the CLI, the API, uploads, Confluence, the web crawler, `db://` and `api://`.
- Trusting identity headers is only safe while callers cannot set them. The service must be reachable only through the API gateway, which strips client-sent identity headers. That restriction is a network-level duty of whoever deploys it.
- The repository no longer provisions infrastructure or deploys itself. It ships a container image (`Dockerfile`). Hosting, secrets storage, scheduling of `scripts/reconcile.py` and `scripts/graph_maintenance.py`, and log and metric collection belong to the platform it runs on. OpenTelemetry metrics are still recorded but are not exported.

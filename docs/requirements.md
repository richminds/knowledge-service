# Requirements & Design Inputs (C-001 – C-015)

Fill this in with the business owner before tuning anything. Many configuration values,
evaluation thresholds and alert limits in this repository are placeholders, and they should
be taken from this page. Each section names the files that depend on it.

> Status: **DRAFT, needs owner input.** Replace every `TODO`.

## 1. Use case (C-001)
<!--  -->
| | |
|---|---|
| Problem | TODO: what people cannot find or answer today |
| Users | TODO: see personas below |
| A good answer | Correct, grounded in internal documents, cites its sources, and says "I don't know" rather than guessing |
| Out of scope | TODO: e.g. legal advice, personal HR cases |

Feeds `ASSISTANT_SCOPE` and `ORGANISATION_NAME` (`src/app/core/config.py`), which drive out-of-scope detection and the prompts.

## 2. Personas (C-002)

| Persona | Typical questions | Access groups | Channel |
|---|---|---|---|
| TODO: e.g. staff member | policies, how-to | `grp-all-staff` | chat UI |
| TODO: e.g. HR advisor | cross-policy, history | `grp-hr` | chat UI |
| TODO: e.g. integration | service-to-service | API key principals | API |

## 3. Sample questions & types (C-003 – C-005)

Collect 50–200 real questions. Label each one and add it to `evals/golden_set.jsonl` (`question_type`).

| Type | Example | Default path (C-005, `RAG_ROUTER_MAP`) |
|---|---|---|
| factoid | "How many days of annual leave do I get?" | semantic |
| relationship / multi-hop | "Which systems depend on the payroll service?" | graph |
| global / thematic | "What are the main themes across audit findings?" | graph (global) |
| multi-step / analytical | "Compare the 2023 and 2024 travel policies and compute the per-diem change" | agentic |
| unanswerable | "What is the CEO's home address?" | must say "not found" |

## 4. Data sources (C-006, C-007)

| Source | Owner | Format | Volume | Change rate | Permission model | Connector |
|---|---|---|---|---|---|---|
| TODO | | | | | | `config/sources.yaml` |

Estimates (C-007), which drive storage, embedding and graph-extraction cost:
- Documents: TODO. Average pages: TODO. Chunks: about documents × pages × 2.
- Embedding tokens for a full index: about chunks × 500. Graph extraction: about chunks × 2,000 tokens with a small model.

## 5. Service levels (C-008)

| Metric | Semantic | Graph | Agentic | Where enforced |
|---|---|---|---|---|
| p95 latency | TODO (e.g. 6 s) | TODO (10 s) | TODO (25 s) | `tests/load/locustfile.py`, monitoring alerts |
| Availability | TODO (99.5 %) | | | App Service plan / slots |
| Freshness | TODO (event-driven, < 15 min) | nightly communities | | ingestion jobs |

## 6. Quality targets (C-009)

Copy the agreed values into `evals/thresholds.yaml`. The pipeline's quality gate enforces them.

| Metric | Target |
|---|---|
| hit / recall@5 | TODO |
| faithfulness (judge) | TODO |
| citation precision | TODO |
| routing accuracy | TODO |

## 7. Cost budget (C-010)

| | Value | Where |
|---|---|---|
| Monthly budget per environment | TODO | `monthlyBudget` in `infra/*.bicepparam` |
| Token alert per hour | TODO | `infra/modules/monitoring.bicep` (`tokenSpike`) |
| Max agent tokens per question | TODO | `AGENTIC_MAX_TOKENS` |

## 8. Compliance (C-011)

| Topic | Decision | Setting |
|---|---|---|
| PII in documents | TODO: mask / block / allow | `PII_MODE`, `PII_ACTION` |
| PII in logs | masked | `LOG_PII_MASKING=true` |
| Data residency | TODO: region | `location` in the pipeline variables |
| Conversation retention | TODO (90 days prod) | Cosmos TTL in `infra/modules/data-services.bicep` |
| Document retention | TODO | `RETENTION_MAX_AGE_DAYS` |
| Audit | every query and ingest is logged | `app.core.audit` |

## 9. Access-control model (C-012)

- Users sign in with Entra ID (`AUTH_MODE=entra`). Their **group object IDs** and **user ID** are the principals.
- Each document carries an `acl` list. This comes from the source (SharePoint permissions, `.acl.json` sidecars, `config/acl_mapping.yaml`) or from the uploader. `"*"` means everyone.
- Every retriever filters by these principals (vector, keyword, graph, agent tools). The LLM never decides access.
- Tenants: TODO. Either one deployment per tenant or one collection per tenant with tenant groups in the ACL.
- Ingestion requires the Entra app role in `INGESTION_REQUIRED_ROLE`.

## 10. Architecture & models (C-013 – C-015)

See [architecture.md](architecture.md) and the [ADRs](adr/).

| Purpose | Model (deployment setting) |
|---|---|
| Answers | TODO, e.g. gpt-4o (`AZURE_OPENAI_CHAT_DEPLOYMENT`) |
| Routing, grading, extraction | TODO, e.g. gpt-4o-mini (`AZURE_OPENAI_UTILITY_DEPLOYMENT`) |
| Embeddings shortlist | text-embedding-3-large, text-embedding-3-small. Compare them with `scripts/run_ablation.py` |

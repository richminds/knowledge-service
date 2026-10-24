# Workflow Execution Guide

This guide covers how to run, deploy and operate the knowledge agent, in the order you will need each step. Each section has the commands to run, what happens when you run them, and how to check that it worked.

| # | Workflow | When |
|---|---|---|
| 1 | [Run locally](#1-run-locally) | First day; every development session |
| 2 | [Ingest documents](#2-ingest-documents) | Loading or updating content (locally or in an environment) |
| 3 | [Ask questions: what happens at runtime](#3-ask-questions-what-happens-at-runtime) | Understanding or debugging answers |
| 4 | [Fill in the placeholders](#4-fill-in-the-placeholders) | Before the first QA deployment |
| 5 | [First deployment](#5-first-deployment) | Once per environment |
| 6 | [Everyday change: commit to production](#6-everyday-change-commit-to-production) | Every change |
| 7 | [Evaluate and promote a RAG mode](#7-evaluate-and-promote-a-rag-mode) | Choosing the production strategy |
| 8 | [Scheduled jobs](#8-scheduled-jobs) | Understanding what should run on its own |
| 9 | [Operations calendar](#9-operations-calendar) | Weekly and monthly |
| 10 | [Troubleshooting](#10-troubleshooting) | When something fails |

---

## 1. Run locally

**Goal:** the API is running on your machine, it has the sample documents indexed, and it answers questions.

| Step | Command | What happens | Check |
|---|---|---|---|
| 1.1 Install | `python -m venv .venv`, then `.venv\Scripts\activate` and `make install` | Installs the app, test and lint tools, and the pre-commit hooks | `python -c "import app"` with `src` on the path, or `make test` passes |
| 1.2 Reach the LLM gateway | Run the API gateway and llm-gateway locally, or use a shared one | Every chat and embedding call goes there. Locally (`AUTH_MODE=none`) there is no caller token, so calls carry `LLM_GATEWAY_API_KEY` as `X-API-Key`. | — |
| 1.3 Configure | `copy .env.example .env`, then set `API_GATEWAY_URL` (or `LLM_GATEWAY_URL`), `LLM_GATEWAY_API_KEY` and `MONGODB_URL` (an Atlas cluster that allows your IP) | The app reads `.env` at startup. Everything else has a local default. `EMBEDDING_DIMENSIONS` must match the gateway's embedding model. | — |
| 1.4 MongoDB indexes | `make create-index` (optional: the API also does this on startup) | Creates the collections, regular indexes and Atlas Search indexes in `MONGODB_DATABASE` | The Atlas UI shows the search indexes as `READY` |
| 1.5 Start the API | `make dev` | Uvicorn with reload on port 8090 (the port api-gateway routes /api/knowledge to). On startup it creates the search index and, if enabled, the graph schema. | `curl localhost:8090/api/v1/health/ready` returns `200` and lists no configuration errors |
| 1.6 Load sample data | `make ingest` | Runs `scripts/ingest_cli.py` on `data/sample/` into the vector and graph stores | The output shows the documents indexed and the chunk and entity counts |
| 1.7 Ask | Open http://localhost:8090/ui | The chat page streams the answer with its steps, sources and confidence | You get an answer with `[1]`-style citations |

**Running everything in containers:** `make docker-run` builds the image and starts the API (MongoDB stays in Atlas). The gateway URL in `.env` must be reachable from inside the container.

**Running only the semantic mode:** set `RAG_ENABLED_MODES=semantic` and `INGESTION_TARGETS=vector`, and skip step 1.4.

---

## 2. Ingest documents

### 2.1 Ways to start ingestion

| Trigger | Command / action | Best for |
|---|---|---|
| CLI | `python scripts/ingest_cli.py --source <uri> --collection <name> [--targets vector,graph] [--full-sync] [--force]` | Local work, backfills, re-indexing |
| API job | `POST /api/v1/ingestion {"source_uri": "...", "collection": "...", "acl": [...]}`, then `GET /api/v1/ingestion/{job_id}` | Ad-hoc loads from another system |
| Upload | `POST /api/v1/ingestion/upload` (multipart: `file`, `acl`, `collection`) | Single files |
| Scheduled reconcile | `scripts/reconcile.py` for the sources in `config/sync_schedule.yaml`, run on a schedule (section 8) | Catching changes, deletions and permission changes |

The CLI and the scheduled scripts run with no caller token, so their LLM calls need `LLM_GATEWAY_API_KEY`. API jobs and uploads run as the calling user.

Source URIs:
- a local path under `INGESTION_LOCAL_ROOT`
- `confluence://<SPACE>`
- `db://<name>` and `api://<name>`, defined in `config/sources.yaml`
- `https://…` for the web crawler, which needs `WEB_CRAWL_ENABLED=true`

### 2.2 What happens to each document

```text
connector ─► raw checks ─► land raw ─► parse ─► clean & govern ─► metadata ─► change detection
   │           size, type,     zone      PDF/Office/HTML/   PII, sensitivity,    title,    unchanged → skip
   │           malware tag,    raw/      email/ZIP/code/    injection scan,      dates,    ACL-only → update ACL
   │           retention                 text, JSON, CSV    duplicates           doc type  changed → continue
   ▼
chunk (+ parents) ─► enrich (optional) ─► embed (cached) ─► index in MongoDB ─► graph extraction ─► manifest
 structure-aware      summary, context,                      new version first,     entities,           status +
 token-sized          questions                              old version pruned     relations, claims   reasons
```

- **Quarantined documents** are kept in the `quarantine/` zone with a reason. A document is quarantined when it is too large, unreadable, low quality, too sensitive, suspected of prompt injection, or flagged by a malware scan.
- **Images and diagrams are not ingested.** PDFs are parsed as text only. See [2.2.1](#221-images-and-diagrams).
- **Re-ingesting is safe.** Unchanged documents are skipped, using the content hash in the manifest. A changed document is written as a new version before the old one is removed, so it never disappears from search during an update.
- **Permissions travel with the document:**
  - Confluence restrictions are read from the source.
  - Local files use a `<file>.acl.json` sidecar.
  - API callers can pass an `acl` explicitly.
  - Principals are auth-service user IDs, account IDs or `*`. Source groups (Confluence) are mapped to them in `config/acl_mapping.yaml`. Unmapped groups fail closed.
  - Documents with no ACL get `INGESTION_DEFAULT_ACL`.

### 2.2.1 Images and diagrams

The LLM gateway takes text-only messages, so there is no OCR, vision or speech model, and the parser reads text only:

| Content | What happens |
|---|---|
| PDF text | `pypdf` text layer |
| Picture inside a PDF (diagram, chart, screenshot) | **Ignored** |
| Image-only (scanned) PDF page | **No text**: the document gets the `pages_without_text:N/M` quality flag |
| Image, audio or video file | **Rejected** with "images and audio/video are not ingested". The upload endpoint does not accept these types. |

**Limits:**
- **Diagrams drawn in the PDF itself** (Visio and PowerPoint exports: lines, text and small logos) keep their text labels, but the connections between boxes are lost.
- To make a diagram or scan answerable, provide its content as text: a text-based PDF, or a document that describes it.

**Checking a document:**
- **Parsed text:** with `ZONES_STORE=local`, open `.ragdata/parsed/<collection>/<doc id>/document.md`.
- **Quarantine reasons:** the quality flags (such as `pages_without_text:N/M`) appear in the manifest's `reasons` only when the document was quarantined for low parse quality.

### 2.3 Verify an ingest

```bash
# Per-document status: indexed / skipped / quarantined / failed, with reasons
curl "localhost:8090/api/v1/ingestion/manifest?collection=default&status=quarantined"
# Index counts against the manifest (C-139)
python scripts/reconcile.py
```

### 2.4 After a large ingest into the graph

```bash
python scripts/graph_maintenance.py all --collection default
```

This merges duplicate entities, embeds entity descriptions, builds communities and their reports (needed for "global" questions), and prints a quality report. In an environment, schedule it nightly (section 8).

### 2.5 Remove content

| Need | Call |
|---|---|
| One document, everywhere (vectors, graph, manifest, cache) | `DELETE /api/v1/ingestion/sources?source=<source>` |
| A whole collection | `DELETE /api/v1/ingestion/collections/<name>` |
| Documents that disappeared from a source | Ingest with `--full-sync` (the scheduled reconcile does this) |

---

## 3. Ask questions: what happens at runtime

`POST /api/v1/chat` (or `/chat/stream`, which sends the same steps as `step` events) runs these steps in order:

| # | Step | Code | Notes |
|---|---|---|---|
| 1 | Authenticate, rate limit | `core/auth.py`, `core/caller_context.py`, `core/ratelimit.py` | Reads the identity headers from the API gateway and builds the caller's principals: user ID, account ID, `*`. Binds the bearer token for the LLM calls. |
| 2 | Validate and safety-check the question | `models/schemas.py`, `rag/safety.py` | A prompt-injection attempt returns `answer_type: blocked` |
| 3 | Load history | `services/conversations.py` | By `conversation_id`, for this user only |
| 4 | Understand the question | `rag/query/analysis.py` | One LLM call works out the standalone question, intent (`rag`, `chitchat`, `out_of_scope`), complexity and filters |
| 5 | Shortcut | `services/retrieval_service.py` | Chit-chat and out-of-scope questions are answered without retrieval |
| 6 | Answer cache | `rag/cache.py` | A similar earlier question with the same permissions returns the cached answer (`cached: true`) |
| 7 | Choose the strategy | `rag/strategies/registry.py` | Uses the request's `rag_mode` (if allowed), or `RAG_MODE`. `auto` picks by complexity. |
| 8 | Retrieve | `rag/strategies/{semantic,graph,agentic}.py` | ACL filters are applied inside every store query. The agent may ask a clarifying question instead. |
| 9 | Flag and assemble context | `rag/safety.py`, `rag/context/` | Injected passages are flagged, then fused, deduplicated, reranked, ordered and fitted to `CONTEXT_MAX_TOKENS` |
| 10 | Generate | `rag/generation/` | Sends the versioned prompt with numbered `[n]` citations to the LLM gateway, which picks the model and handles failover |
| 11 | Check the output | `rag/safety.py` | Leaked PII is withheld |
| 12 | Assess quality | `rag/evaluation/` | Signals, plus the judge when requested or sampled, give the confidence |
| 13 | Escalate (optional) | `services/retrieval_service.py` | With low confidence and `RAG_ESCALATION_ENABLED`, steps 8–12 are retried with a stronger mode |
| 14 | Finish | — | Saves the turn, caches the answer, and writes the audit entry, the `rag_answer` log and metrics |

To see the steps for one request, add `"include_trace": true`. This needs `RAG_ALLOW_TRACE`, which is on in QA. In the logs, every JSON line of the request carries `request_id`, which is the response's `response_id`; filter on it.

---

## 4. Fill in the placeholders

Do this before the first QA deployment. Every item is marked `TODO` in the repository, so `grep -rn TODO config src evals docs` finds them all.

| Area | File | What to set |
|---|---|---|
| Business inputs | `docs/requirements.md` | Use case, personas, service levels, quality targets, budget, compliance, ACL model |
| Assistant identity | `src/app/core/config.py` or environment variables | `ORGANISATION_NAME`, `ASSISTANT_SCOPE` |
| Gateways | Environment variables | `API_GATEWAY_URL` (or `LLM_GATEWAY_URL`), `LLM_GATEWAY_API_KEY`, `EMBEDDING_DIMENSIONS` |
| Sign-in | Environment variables | `AUTH_MODE=gateway`, `ADMIN_REQUIRES_ADMIN`, `INGESTION_REQUIRES_ADMIN` |
| Sources | `config/sources.yaml`, `config/sync_schedule.yaml` | Databases, APIs and the nightly sync list |
| Permissions | `config/acl_mapping.yaml` | Source group → auth-service user or account ID |
| Search vocabulary | `config/glossary.yaml`, `config/metadata_rules.yaml` | Acronyms and synonyms; rules for document type and department |
| Graph | `config/ontology.yaml`, `config/entity_aliases.yaml`, `config/cypher_templates.yaml` | Entity and relation types, aliases, query templates |
| Evaluation | `evals/golden_set.jsonl`, `evals/thresholds.yaml` | Real questions reviewed by experts; minimum scores |
| Connectors (if used) | Environment variables | `CONFLUENCE_API_TOKEN`, `WEB_CRAWL_AUTH_HEADER`, and any `${ENV_VAR}` referenced in `config/sources.yaml` |

---

## 5. First deployment

Do this once per environment (QA first, then Prod). The repository ships a container image (`Dockerfile`); it does not provision infrastructure or deploy itself.

### 5.1 Prerequisites (one-time, outside this repo)

1. **API gateway and llm-gateway** running in the environment, with `/api/llm` routed to llm-gateway. The gateway's embedding model sets `EMBEDDING_DIMENSIONS`.
2. **An llm-gateway service key** (`LLM_GATEWAY_API_KEY`) if the CLI or the scheduled scripts run against this environment.
3. **MongoDB Atlas**: a cluster, a database user, and a network access list that allows the hosts running the service and its scripts.
4. **Redis** (`REDIS_URL`), if several instances must share the rate limit or the answer cache.
5. **Network restriction**: only the API gateway may reach the service. `AUTH_MODE=gateway` trusts the identity headers on every request it receives.

### 5.2 Build and run

```bash
docker build -t oissa-rag-agent:<tag> .
```

Run the image with the environment's settings as environment variables: start from the production profile in [feature-flags.md, section 4](feature-flags.md#4-ready-made-profiles), and add `API_GATEWAY_URL`, `LLM_GATEWAY_API_KEY`, `MONGODB_URL` and the connector secrets. The container listens on `PORT` (8000). Keep `WEB_CONCURRENCY=1` while ingestion job status lives in memory.

### 5.3 After the first deployment

| Step | Command / action | Why |
|---|---|---|
| Check readiness | `curl <service URL>/api/v1/health/ready` | Must list no configuration errors |
| Check sign-in | Call `/api/v1/chat` through the API gateway, then directly without `X-User-ID` | Through the gateway it answers; directly it returns `401` (or should be unreachable) |
| Load content | Run `ingest_cli.py` with the environment's settings, or use the ingestion API through the gateway | Fills the index and graph |
| Build graph communities | `python scripts/graph_maintenance.py all` | Needed for global questions |
| Schedule the jobs | Section 8 | Keeps sources and the graph current |
| Record a baseline | `python scripts/run_eval.py --rag-mode all --judge --save-baseline`, then commit `evals/results/baseline.json` | Later `--gate` runs compare against it |

---

## 6. Everyday change: commit to production

1. **Branch** from `develop`: `feature/<name>`.
2. **Before pushing**, run locally:
   ```bash
   make format && make check            # lint, types, tests
   python scripts/run_eval.py --rag-mode all --judge   # if prompts, retrieval or chunking changed
   ```
3. **Open a PR to `develop`**, and merge it after review.
4. **Deploy to QA**: build the image from `develop` and run it (5.2). Run the quality gate against QA: `python scripts/run_eval.py --rag-mode all --judge --gate`. Test with `/ui`, or with `rag_mode` overrides.
5. **Merge `develop` into `main`**, build the image from `main` and deploy it to Prod.
6. **Watch** the logs, readiness and 👎 feedback for about 30 minutes. If something is wrong, redeploy the previous image ([rollback runbook](runbooks/rollback.md)).

Changes that need more than a deploy:

| Change | Extra step |
|---|---|
| The gateway's embedding model, index schema, chunking or parsers | [Full re-index](runbooks/reindex.md) |
| New prompt version | Add it to `src/app/prompts/*.yaml`, evaluate it, then set `PROMPT_VERSIONS` |
| New secret | Set the environment variable in each environment, then restart the service |

---

## 7. Evaluate and promote a RAG mode

Production starts on `semantic`. To move to `graph`, `agentic` or `auto`:

1. **Ingest into both stores in QA** with `INGESTION_TARGETS=vector,graph`. Then run graph maintenance (2.4).
2. **Compare on the golden set:**
   ```bash
   python scripts/run_eval.py --rag-mode all --judge       # per mode and per question type
   python scripts/run_eval.py --rag-mode auto              # routing accuracy
   python scripts/run_ablation.py                          # optional: tuning variants
   ```
3. **Check cost and latency.** The eval report shows latency and tokens per mode. The LLM gateway's per-caller stats (`LLM_CALLER:<purpose>`) show tokens by purpose.
4. **Decide.** Choose the cheapest mode, or `auto`, that meets the quality targets in `docs/requirements.md`. Record the decision in an ADR.
5. **Promote.** In Prod's environment variables:
   - set `RAG_MODE` and add the mode to `RAG_ENABLED_MODES`;
   - set `INGESTION_TARGETS` to include `graph` if needed;
   - optionally set `RAG_ESCALATION_ENABLED=true`.
6. **Backfill Prod.** If graph is new there, ingest into the graph with `--targets graph --force`, then run graph maintenance.
7. **Restart** the service, and compare the eval results and feedback for a week.

---

## 8. Scheduled jobs

Nothing runs on its own until you schedule it. Run these with the image's `scripts/` (or a checkout) and the environment's settings, from any scheduler (cron or similar). They have no caller token, so they need `LLM_GATEWAY_API_KEY`.

| Job | Suggested trigger | Runs | Purpose |
|---|---|---|---|
| Reconcile | Daily | `python scripts/reconcile.py` | Full sync of `config/sync_schedule.yaml`: new, changed and deleted documents, ACL changes, count check |
| Graph maintenance | Daily, after reconcile (graph enabled) | `python scripts/graph_maintenance.py all` | Entity resolution, embeddings, communities and reports, quality report |

- **Failures:** `reconcile.py` exits with code 1 when any source fails. Check the scheduler's run history and the script output, and `GET /api/v1/ingestion/manifest?status=failed`.

---

## 9. Operations calendar

| Cadence | Task | Guide |
|---|---|---|
| Daily (scheduled) | Reconcile, graph maintenance | Section 8 |
| Weekly | Triage 👎 feedback, add failures to the golden set, review unanswered topics | [operations.md](runbooks/operations.md) |
| Monthly | Retune with ablations, review cost, report to stakeholders | [operations.md](runbooks/operations.md) |
| On a gateway model change | Re-evaluate, update the ADR | [operations.md](runbooks/operations.md) |
| Quarterly | Calibrate the judge; red team | [red-team.md](runbooks/red-team.md) |
| On demand | Re-index, roll back, handle incidents | [reindex.md](runbooks/reindex.md), [rollback.md](runbooks/rollback.md), [incidents.md](runbooks/incidents.md) |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health/ready` lists configuration errors | A required setting is missing for an enabled mode, or no gateway URL is set | Set it in `.env` or the environment, or disable that mode |
| `401` on every call | `AUTH_MODE=gateway` and the request has no `X-User-ID` | Call through the API gateway; locally use `AUTH_MODE=none` |
| `403` on ingestion or the admin studio | `INGESTION_REQUIRES_ADMIN` or `ADMIN_REQUIRES_ADMIN` is on and the caller is not an admin (`X-Is-Admin`) | Use an admin account, or turn the setting off |
| LLM calls fail with `401`/`403` from the gateway | No caller token and no `LLM_GATEWAY_API_KEY` (CLI, scripts, `AUTH_MODE=none`), or the key is wrong | Set `LLM_GATEWAY_API_KEY` |
| `400` "RAG mode … is not enabled" | The mode is not in `RAG_ENABLED_MODES`, or override is off | Enable the mode, or drop `rag_mode` from the request |
| Answers say "not found" for indexed content | The ACL doesn't match the user's ID or account ID; filters are too narrow; or the document was quarantined | Check the manifest status and the document's `acl`, and retry with `include_trace` |
| Questions about a diagram, chart or scanned page get "not found" | PDFs are parsed as text only | Provide the content as text and re-ingest it ([2.2.1](#221-images-and-diagrams)) |
| Image or audio upload fails | These types are not ingested | Convert the content to text |
| Vector search errors or returns nothing after a gateway change | The gateway's embedding model no longer matches `EMBEDDING_DIMENSIONS` | Set `EMBEDDING_DIMENSIONS` to match and [re-index](runbooks/reindex.md) |
| Graph mode returns nothing | No entities extracted, no communities yet, or the entity vector index is still building | Check readiness; ingest with the `graph` target; run graph maintenance |
| MongoDB timeouts or `operation cancelled` | The Atlas network access list doesn't allow the caller's IP, or a TLS-decrypting proxy breaks authentication | Allow the outbound IPs in Atlas; exempt `*.mongodb.net:27017` from TLS inspection |
| `429` from the LLM gateway | The user's or the service's rate limit or budget there is exhausted | Raise the limit in the gateway, or lower the agent limits and `QUALITY_JUDGE_SAMPLE_RATE` |
| Quality gate fails | Scores fell below `evals/thresholds.yaml` or regressed from the baseline | Read the run's report, fix the cause, or update the baseline intentionally |

Still stuck? See the [incident runbook](runbooks/incidents.md).

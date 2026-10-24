# Workflow Execution Guide

This guide covers how to run, deploy and operate the knowledge agent, in the order you will need each step. Each section has the commands to run, what happens when you run them, and how to check that it worked.

| # | Workflow | When |
|---|---|---|
| 1 | [Run locally](#1-run-locally) | First day; every development session |
| 2 | [Ingest documents](#2-ingest-documents) | Loading or updating content (locally or in Azure) |
| 3 | [Ask questions: what happens at runtime](#3-ask-questions-what-happens-at-runtime) | Understanding or debugging answers |
| 4 | [Fill in the placeholders](#4-fill-in-the-placeholders) | Before the first QA deployment |
| 5 | [First deployment to Azure](#5-first-deployment-to-azure) | Once per environment |
| 6 | [Everyday change: commit to production](#6-everyday-change-commit-to-production) | Every change |
| 7 | [Evaluate and promote a RAG mode](#7-evaluate-and-promote-a-rag-mode) | Choosing the production strategy |
| 8 | [Scheduled and background jobs](#8-scheduled-and-background-jobs) | Understanding what runs on its own |
| 9 | [Operations calendar](#9-operations-calendar) | Weekly and monthly |
| 10 | [Troubleshooting](#10-troubleshooting) | When something fails |

---

## 1. Run locally

**Goal:** the API is running on your machine, it has the sample documents indexed, and it answers questions.

| Step | Command | What happens | Check |
|---|---|---|---|
| 1.1 Install | `python -m venv .venv`, then `.venv\Scripts\activate` and `make install` | Installs the app, test and lint tools, and the pre-commit hooks | `python -c "import app"` with `src` on the path, or `make test` passes |
| 1.2 Sign in to Azure | `az login` | Your Entra ID account is used for Azure OpenAI, so no key is needed | `az account show` |
| 1.3 Configure | `copy .env.example .env`, then set `AZURE_OPENAI_ENDPOINT` and `MONGODB_URL` (an Atlas cluster that allows your IP) | The app reads `.env` at startup. Everything else has a local default. | — |
| 1.4 MongoDB indexes | `make create-index` (optional: the API also does this on startup) | Creates the collections, regular indexes and Atlas Search indexes in `MONGODB_DATABASE` | The Atlas UI shows the search indexes as `READY` |
| 1.5 Start the API | `make dev` | Uvicorn with reload on port 8000. On startup it creates the search index and, if enabled, the graph schema. | `curl localhost:8000/api/v1/health/ready` returns `200` and lists no configuration errors |
| 1.6 Load sample data | `make ingest` | Runs `scripts/ingest_cli.py` on `data/sample/` into the vector and graph stores | The output shows the documents indexed and the chunk and entity counts |
| 1.7 Ask | Open http://localhost:8000/ui | The chat page streams the answer with its steps, sources and confidence | You get an answer with `[1]`-style citations |

**Your Azure account needs these roles:**
- *Cognitive Services OpenAI User* on the OpenAI resource.
- *Search Index Data Contributor* and *Search Service Contributor* on the search service.
- *Key Vault Secrets User*, only if you use `AZURE_KEY_VAULT_URL`.

**Running everything in containers:** `make docker-run` builds the image and starts the API (MongoDB stays in Atlas). It reuses your `~/.azure` login inside the container.

**Running only the semantic mode:** set `RAG_ENABLED_MODES=semantic` and `INGESTION_TARGETS=vector`, and skip step 1.4.

---

## 2. Ingest documents

### 2.1 Ways to start ingestion

| Trigger | Command / action | Best for |
|---|---|---|
| CLI | `python scripts/ingest_cli.py --source <uri> --collection <name> [--targets vector,graph] [--full-sync] [--force]` | Local work, backfills, re-indexing |
| API job | `POST /api/v1/ingestion {"source_uri": "...", "collection": "...", "acl": [...]}`, then `GET /api/v1/ingestion/{job_id}` | Ad-hoc loads from another system |
| Upload | `POST /api/v1/ingestion/upload` (multipart: `file`, `acl`, `collection`) | Single files |
| Event-driven (Azure) | Upload to the storage container `documents` | Day-to-day updates; nothing to run |
| Nightly reconcile (Azure) | Automatic at 02:00 UTC for the sources in `config/sync_schedule.yaml` | Catching changes, deletions and permission changes |

Source URIs:
- a local path under `INGESTION_LOCAL_ROOT`
- `blob://container/prefix`
- `sharepoint://<host>/sites/<site>/<library>/<folder>`
- `confluence://<SPACE>`
- `db://<name>` and `api://<name>`, defined in `config/sources.yaml`
- `https://…` for the web crawler, which needs `WEB_CRAWL_ENABLED=true`

### 2.2 What happens to each document

```text
connector ─► raw checks ─► land raw ─► parse ─► clean & govern ─► metadata ─► change detection
   │           size, type,     zone      PDF/Office/HTML/   PII, sensitivity,    title,    unchanged → skip
   │           malware tag,    raw/      email/ZIP/code/    injection scan,      dates,    ACL-only → update ACL
   │           retention                 images/audio       duplicates           doc type  changed → continue
   ▼
chunk (+ parents) ─► enrich (optional) ─► embed (cached) ─► index in MongoDB ─► graph extraction ─► manifest
 structure-aware      summary, context,                      new version first,     entities,           status +
 token-sized          questions                              old version pruned     relations, claims   reasons
```

- **Quarantined documents** are kept in the `quarantine/` zone with a reason. A document is quarantined when it is too large, unreadable, low quality, too sensitive, suspected of prompt injection, or flagged by a malware scan.
- **Images and diagrams** are handled by two services, both optional. See [2.2.1](#221-images-and-diagrams) for the details and limits.
  - **Document Intelligence** reads text (OCR), tables and layout.
  - **The vision deployment** (`AZURE_OPENAI_VISION_DEPLOYMENT`, e.g. `gpt-5.5`) describes what a picture shows: components, connections, chart values.
- **Re-ingesting is safe.** Unchanged documents are skipped, using the content hash in the manifest. A changed document is written as a new version before the old one is removed, so it never disappears from search during an update.
- **Permissions travel with the document:**
  - SharePoint permissions and Confluence restrictions are read from the source.
  - Blob and local files use `acl` blob metadata or a `<file>.acl.json` sidecar.
  - API callers can pass an `acl` explicitly.
  - Group names without an Entra ID are mapped in `config/acl_mapping.yaml`. Unmapped groups fail closed.
  - Documents with no ACL get `INGESTION_DEFAULT_ACL`.

### 2.2.1 Images and diagrams

What happens depends on the file and on which services are configured:

| Content | Document Intelligence (`DOC_INTELLIGENCE_ENDPOINT`) | Vision deployment (`AZURE_OPENAI_VISION_DEPLOYMENT`) | Neither |
|---|---|---|---|
| PDF text | Layout model: reading order, headings, tables | `pypdf` text layer | `pypdf` text layer |
| Picture inside a PDF (diagram, chart, screenshot) | Caption and any text inside it; not what it shows | Described and indexed as `Figure (page N): …` on its page | **Ignored** |
| Image-only (scanned) PDF page | OCR text | Described, including any visible text | **No text**: the manifest shows `pages_without_text` |
| Image file (PNG, JPEG, BMP, TIFF, HEIF) | OCR text. With `IMAGE_PARSER=auto`, a vision description is added when OCR finds under `IMAGE_MIN_OCR_CHARS` characters. | Described | **Fails**, with a reason naming the setting |
| GIF or WebP file | Not supported | Described | **Fails** |

**Settings** (details in [feature-flags.md](feature-flags.md#3-ingestion-workflow-and-flags)):
- `IMAGE_PARSER`: `auto`, `docintel`, `vision`, `docintel+vision` or `off`.
- `PDF_IMAGE_DESCRIPTIONS`: on by default. Takes effect when the vision deployment is set and the PDF is parsed with `pypdf`.
- `PDF_IMAGE_MIN_PIXELS` (150): images narrower than this on their shorter side are skipped as icons or logos. An image that repeats, such as a logo in every page header, is described once.
- `PDF_MAX_IMAGES` (30): images described per PDF. The rest are flagged as `pdf_images_not_described:N`.
- `VISION_CONCURRENCY` (4) and `VISION_MAX_TOKENS` (1500): images described at once, and the length of each description.

**Supported vision models:** gpt-4o, gpt-4.1, and the gpt-5.x / o-series reasoning models. Reasoning models are detected by name, and a deployment with a custom name is switched to reasoning parameters after its first rejection.

**Cost and time:** each image takes about 15–20 s and about 3,000 tokens on `gpt-5.5`. This is paid once, at ingestion. Asking questions isn't affected.

**Limits:**
- **Diagrams drawn in the PDF itself are not described.** Visio and PowerPoint exports are usually lines, text and small logos rather than one picture. Their text labels are indexed, but there's no single image to send to the vision model, so the connections between boxes are lost. Rendering such pages as images is not built yet (see [Known gaps](../README.md#known-gaps)).
- **A description can lose which label belongs to which arrow.** Answers about a specific connection, such as "which port does X use", should be checked against the diagram.
- **When Document Intelligence parses a PDF, the vision model is not used for its pictures.** Only captions and text inside figures are kept.

**Turning it on in Azure:** in `infra/<env>.bicepparam`, add `param openAiVisionDeployment = '<deployment>'`. Optionally, set `enableDocIntelligence = true` for OCR and tables. Then re-ingest the affected documents with `--force`.

**Checking a document:**
- **Logs:** the `pdf_images_described` event shows, per PDF, how many images were extracted, described and over the limit. `pdf_image_description_failed` names the page of any image that failed.
- **Parsed text:** with `ZONES_STORE=local`, open `.ragdata/parsed/<collection>/<doc id>/document.md`. Each `Figure (page N): …` block is what the vision model wrote. In Azure it's the same path in the `rag-zones` container.
- **Quarantine reasons:** the quality flags (`pages_without_text:N/M`, `pdf_images_not_described:N`, `pdf_image_descriptions_failed:N`) appear in the manifest's `reasons` only when the document was quarantined for low parse quality.

### 2.3 Verify an ingest

```bash
# Per-document status: indexed / skipped / quarantined / failed, with reasons
curl "localhost:8000/api/v1/ingestion/manifest?collection=default&status=quarantined"
# Index counts against the manifest (C-139)
python scripts/reconcile.py
```

### 2.4 After a large ingest into the graph

```bash
python scripts/graph_maintenance.py all --collection default
```

This merges duplicate entities, embeds entity descriptions, builds communities and their reports (needed for "global" questions), and prints a quality report. In Azure it runs nightly at 03:00 UTC.

### 2.5 Remove content

| Need | Call |
|---|---|
| One document, everywhere (vectors, graph, manifest, cache) | `DELETE /api/v1/ingestion/sources?source=<source>` |
| A whole collection | `DELETE /api/v1/ingestion/collections/<name>` |
| Documents that disappeared from a source | Ingest with `--full-sync` (the nightly reconcile does this) |
| A blob deleted from `documents` | Automatic: a `BlobDeleted` event reaches the queue and the worker deletes it |

---

## 3. Ask questions: what happens at runtime

`POST /api/v1/chat` (or `/chat/stream`, which sends the same steps as `step` events) runs these steps in order:

| # | Step | Code | Notes |
|---|---|---|---|
| 1 | Authenticate, rate limit | `core/auth.py`, `core/ratelimit.py` | Builds the caller's principals: user ID, Entra groups, `*` |
| 2 | Validate and safety-check the question | `models/schemas.py`, `rag/safety.py` | An injection or harmful content returns `answer_type: blocked` |
| 3 | Load history | `services/conversations.py` | By `conversation_id`, for this user only |
| 4 | Understand the question | `rag/query/analysis.py` | One utility-model call works out the standalone question, intent (`rag`, `chitchat`, `out_of_scope`), complexity and filters |
| 5 | Shortcut | `services/retrieval_service.py` | Chit-chat and out-of-scope questions are answered without retrieval |
| 6 | Answer cache | `rag/cache.py` | A similar earlier question with the same permissions returns the cached answer (`cached: true`) |
| 7 | Choose the strategy | `rag/strategies/registry.py` | Uses the request's `rag_mode` (if allowed), or `RAG_MODE`. `auto` picks by complexity. |
| 8 | Retrieve | `rag/strategies/{semantic,graph,agentic}.py` | ACL filters are applied inside every store query. The agent may ask a clarifying question instead. |
| 9 | Flag and assemble context | `rag/safety.py`, `rag/context/` | Injected passages are flagged, then fused, deduplicated, reranked, ordered and fitted to `CONTEXT_MAX_TOKENS` |
| 10 | Generate | `rag/generation/` | Uses the versioned prompt and numbered `[n]` citations, failing over to the backup model on errors |
| 11 | Check the output | `rag/safety.py` | Leaked PII or harmful output is withheld |
| 12 | Assess quality | `rag/evaluation/` | Signals, plus the judge when requested or sampled, give the confidence |
| 13 | Escalate (optional) | `services/retrieval_service.py` | With low confidence and `RAG_ESCALATION_ENABLED`, steps 8–12 are retried with a stronger mode |
| 14 | Finish | — | Saves the turn, caches the answer, and writes the audit entry, the `rag_answer` log and metrics |

To see the steps for one request, add `"include_trace": true`. This needs `RAG_ALLOW_TRACE`, which is on in QA. In Azure, look the request up by `response_id`:

```kusto
traces | where customDimensions.request_id == "<response_id>" | order by timestamp asc
```

---

## 4. Fill in the placeholders

Do this before the first QA deployment. Every item is marked `TODO` in the repository, so `grep -rn TODO config infra src evals docs` finds them all.

| Area | File | What to set |
|---|---|---|
| Business inputs | `docs/requirements.md` | Use case, personas, service levels, quality targets, budget, compliance, ACL model |
| Assistant identity | `src/app/core/config.py` or app settings | `ORGANISATION_NAME`, `ASSISTANT_SCOPE` |
| Azure OpenAI | `infra/qa.bicepparam`, `infra/prod.bicepparam` | `openAiAccountName` (+ `openAiResourceGroup`), deployment names |
| Container registry | `infra/*.bicepparam`, `pipelines/variables/common.yml` | `containerRegistryName`, `containerRegistry` |
| Pipeline | `pipelines/variables/*.yml` | Service connection names, resource groups, region |
| Sign-in | `infra/*.bicepparam` | `authMode`, `entraTenantId`, `entraApiAudience`, `ingestionRequiredRole` |
| Operations | `infra/*.bicepparam` | `alertEmails`, `monthlyBudget`, `enable*` flags; resource `tags` (owner, cost-center) in `main.bicep` |
| Sources | `config/sources.yaml`, `config/sync_schedule.yaml` | Databases, APIs and the nightly sync list |
| Permissions | `config/acl_mapping.yaml` | Source group → Entra group object ID |
| Search vocabulary | `config/glossary.yaml`, `config/metadata_rules.yaml` | Acronyms and synonyms; rules for document type and department |
| Graph | `config/ontology.yaml`, `config/entity_aliases.yaml`, `config/cypher_templates.yaml` | Entity and relation types, aliases, query templates |
| Evaluation | `evals/golden_set.jsonl`, `evals/thresholds.yaml` | Real questions reviewed by experts; minimum scores |
| Images and scanned PDFs | `infra/*.bicepparam` | `openAiVisionDeployment` (describes diagrams and pictures; not set yet in either file), `enableDocIntelligence`, `imageParser`, `docIntelligenceImageModel`. See [2.2.1](#221-images-and-diagrams). |
| Connectors (if used) | Key Vault | `sharepoint-client-secret`, `confluence-api-token`, and the other connector secrets |

---

## 5. First deployment to Azure

Do this once per environment (QA first, then Prod).

### 5.1 Prerequisites (one-time, outside this repo)

1. **Azure OpenAI**:
   - Create the account with deployments for chat, utility, embedding and (optionally) vision/Whisper.
   - Check you have quota for both environments.
2. **Container registry**: an existing ACR, named in `pipelines/variables/common.yml`.
3. **Entra app registration** (when `authMode` is `entra`):
   - Expose an API and note the App ID URI (`api://…`). That becomes `entraApiAudience`.
   - Add app roles, for example `Knowledge.Ingest` for `ingestionRequiredRole`.
   - Under *Token configuration*, add the **groups** claim so document ACLs can match the user's groups.
4. **Azure DevOps**:
   - **Service connections:** a Docker Registry connection to the ACR, and one ARM connection per environment. Workload identity federation is recommended. Give it *Owner*, or *Contributor + User Access Administrator*, on the resource group, because Bicep creates role assignments. If it cannot create resource groups, create `rg-oissa-rag-qa` and `rg-oissa-rag-prod` in advance.
   - **Environments:** create `qa` and `prod`, and add an **Approvals** check on `prod`.
5. **Check the templates** compile: `make bicep-check`, or `az bicep build --file infra/main.bicep`.
6. **Optional dry run:**
   ```bash
   az deployment group what-if -g rg-oissa-rag-qa --parameters infra/qa.bicepparam \
     --parameters containerImage=<acr>.azurecr.io/oissa-rag-agent:<tag>
   ```

### 5.2 Run the pipeline

Push to `develop` (QA) or `main` (QA, then Prod). The pipeline runs the steps below, and the template provisions everything shown in the README's architecture table:

```text
Build ──────► DeployQA ──────────────────────► QualityGate ─────────► DeployProd
lint, tests,   az deployment group create      golden set vs QA        approval → Bicep →
Bicep compile, (infra/main.bicep + modules)    (run_eval.py --gate)    staging slot → smoke
image push                                                             test → swap
               is pending → deploy container
               → readiness smoke test
```

### 5.3 After the first deployment

| Step | Command / action | Why |
|---|---|---|
| Create the API key secret | `az keyvault secret set --vault-name <kv> --name api-key --value <random>` | Needed when `authMode` includes `api_key` |
| Add connector secrets | `az keyvault secret set …` for each connector you use | Connectors read them at startup |
| Restart the app | `az webapp restart -g <rg> -n <app>` | Picks up the new secrets |
| Check readiness | `curl https://<app>.azurewebsites.net/api/v1/health/ready` | Must list no configuration errors |
| Load content | Upload to the `documents` container (2.1), or run `ingest_cli.py` against the QA settings | Fills the index and graph |
| Build graph communities | Wait for the 03:00 job, or start it: `az containerapp job start -g <rg> -n job-graph-<id>` (name in the portal or `az containerapp job list -g <rg>`) | Needed for global questions |
| Grant the quality gate access | Give the QA service connection's principal the roles listed in `pipelines/templates/eval-gate.yml` | Lets the gate query QA |
| Record a baseline | `python scripts/run_eval.py --rag-mode all --judge --save-baseline`, then commit `evals/results/baseline.json` | The gate compares future runs against it |

Uploading a document with permissions:

```bash
az storage blob upload --auth-mode login --account-name <storage> -c documents \
  -f policy.pdf -n hr/policy.pdf --metadata acl=<entra-group-object-id>
```

---

## 6. Everyday change: commit to production

1. **Branch** from `develop`: `feature/<name>`.
2. **Before pushing**, run locally:
   ```bash
   make format && make check            # lint, types, tests
   make bicep-check                     # only if infra/ changed
   python scripts/run_eval.py --rag-mode all --judge   # if prompts, retrieval or chunking changed
   ```
3. **Open a PR to `develop`.** The pipeline runs **Build** only: lint, tests, Bicep compile and an image build (not pushed).
4. **Merge to `develop`.** The image is pushed, then **DeployQA** and the **QualityGate** run. Test in QA with `/ui`, or with `rag_mode` overrides.
5. **Merge `develop` into `main`.** QA runs again, then **DeployProd** waits for approval.
6. **Approve.** The release goes to the staging slot, is smoke-tested, then swapped into production.
7. **Watch** the workbook and alerts for about 30 minutes. If something is wrong, swap back ([rollback runbook](runbooks/rollback.md)).

Changes that need more than a deploy:

| Change | Extra step |
|---|---|
| Embedding model, index schema, chunking or parsers | [Full re-index](runbooks/reindex.md) |
| New prompt version | Add it to `src/app/prompts/*.yaml`, evaluate it, then set `PROMPT_VERSIONS` |
| New Key Vault secret | Create it in each vault, then restart the app |

---

## 7. Evaluate and promote a RAG mode

Production starts on `semantic`. To move to `graph`, `agentic` or `auto`:

1. **Ingest into both stores in QA.** QA defaults to `ingestionTargets = 'vector,graph'`. Then run graph maintenance (2.4).
2. **Compare on the golden set:**
   ```bash
   python scripts/run_eval.py --rag-mode all --judge       # per mode and per question type
   python scripts/run_eval.py --rag-mode auto              # routing accuracy
   python scripts/run_ablation.py                          # optional: tuning variants
   ```
3. **Check cost and latency.** In the QA workbook, look at the panels for *latency by mode* and *tokens by purpose*.
4. **Decide.** Choose the cheapest mode, or `auto`, that meets the quality targets in `docs/requirements.md`. Record the decision in an ADR.
5. **Promote.** In `infra/prod.bicepparam`:
   - set `ragMode` and add the mode to `ragEnabledModes`;
   - set `ingestionTargets` to include `graph` if needed;
   - optionally set `ragEscalation = true`.
6. **Backfill Prod.** If graph is new there, ingest into the graph with `--targets graph --force`, then run graph maintenance.
7. **Release** through the pipeline (section 6), and compare the workbook for a week.

---

## 8. Scheduled and background jobs

These are deployed when `enableIngestionWorkers = true`. All three jobs run the same container image as the API.

| Job | Trigger | Runs | Purpose |
|---|---|---|---|
| `job-ingest-*` | Service Bus queue `ingestion` has messages (KEDA, up to 5 parallel) | `scripts/ingestion_worker.py --once` | Ingests or deletes the blobs named in Event Grid events, and ingests API job messages |
| `job-reconcile-*` | Daily at 02:00 UTC | `scripts/reconcile.py` | Full sync of `config/sync_schedule.yaml`: new, changed and deleted documents, ACL changes, count check |
| `job-graph-*` | Daily at 03:00 UTC (graph enabled) | `scripts/graph_maintenance.py all` | Entity resolution, embeddings, communities and reports, quality report |

- **Failures:** after 5 failed deliveries, a message moves to the queue's dead-letter queue (see the [incident runbook](runbooks/incidents.md)).
- **Run a job now:** `az containerapp job start -g <rg> -n <job-name>`.
- **See past runs:** `az containerapp job execution list -g <rg> -n <job-name> -o table`.

---

## 9. Operations calendar

| Cadence | Task | Guide |
|---|---|---|
| Daily (automatic) | Reconcile, graph maintenance, alerts | Section 8; `infra/modules/monitoring.bicep` |
| Weekly | Triage 👎 feedback, add failures to the golden set, review unanswered topics | [operations.md](runbooks/operations.md) |
| Monthly | Retune with ablations, review cost, report to stakeholders | [operations.md](runbooks/operations.md) |
| On a new model release | Re-evaluate the models, update the ADR | [operations.md](runbooks/operations.md) |
| Quarterly | Calibrate the judge; red team | [red-team.md](runbooks/red-team.md) |
| On demand | Re-index, roll back, handle incidents | [reindex.md](runbooks/reindex.md), [rollback.md](runbooks/rollback.md), [incidents.md](runbooks/incidents.md) |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health/ready` lists configuration errors | A required setting is missing for an enabled mode | Set it in `.env` or the bicepparam, or disable that mode |
| `401` on every call | `AUTH_MODE` requires a token or key | Send `Authorization: Bearer …` or `X-API-Key`; locally use `AUTH_MODE=none` |
| `403` on ingestion | Caller lacks `INGESTION_REQUIRED_ROLE` | Assign the app role in Entra ID |
| `400` "RAG mode … is not enabled" | The mode is not in `RAG_ENABLED_MODES`, or override is off | Enable the mode, or drop `rag_mode` from the request |
| Answers say "not found" for indexed content | The ACL doesn't match the user's groups; filters are too narrow; or the document was quarantined | Check the manifest status and the document's `acl`, and retry with `include_trace` |
| Questions about a diagram, chart or scanned page get "not found" | No vision deployment or Document Intelligence; the diagram is drawn in the PDF rather than embedded as a picture; or the document was ingested before the service was enabled | Check the `pdf_images_described` log event and the parsed text for `Figure (page N)` blocks ([2.2.1](#221-images-and-diagrams)), configure the service, and re-ingest with `--force` |
| Image upload fails with "No image processor" | Neither service can read that format | Set `AZURE_OPENAI_VISION_DEPLOYMENT` (reads PNG, JPEG, GIF, WebP) or `DOC_INTELLIGENCE_ENDPOINT` (reads PNG, JPEG, BMP, TIFF, HEIF) |
| Graph mode returns nothing | No entities extracted, no communities yet, or the entity vector index is still building | Check readiness; ingest with the `graph` target; run graph maintenance |
| MongoDB timeouts or `operation cancelled` | The Atlas network access list doesn't allow the caller's IP, or a TLS-decrypting proxy breaks authentication | Allow the outbound IPs in Atlas; exempt `*.mongodb.net:27017` from TLS inspection |
| `429` from Azure OpenAI | Quota exceeded | Raise the quota, set the `AZURE_OPENAI_FALLBACK_*` settings, or lower the agent limits |
| Quality gate fails | Scores fell below `evals/thresholds.yaml` or regressed from the baseline | Download the `eval-results` artifact, fix the cause, or update the baseline intentionally |
| Event-driven ingest does nothing | Not uploaded to the `documents` container, the job is failing, or messages are dead-lettered | Check the queue counts and the job executions (section 8) |

Still stuck? See the [incident runbook](runbooks/incidents.md).

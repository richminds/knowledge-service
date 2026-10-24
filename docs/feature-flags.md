# Feature Flags and the Workflow They Control

This guide follows a question and a document through the pipeline and shows, at each step, which setting turns it on or off, what changes, and what it costs. Use it to answer "what happens if I flip this?" before you flip it.

- Every flag is an environment variable, set in `.env` locally or in the service's environment for QA and Prod. The full list with comments is in [.env.example](../.env.example), and the defaults are in [src/app/core/config.py](../src/app/core/config.py).
- **Settings are read once, at startup.** After changing `.env`, restart the API. Files read per request, such as `evals/golden_set.jsonl`, don't need a restart.
- **Ingestion flags only affect documents ingested after the change.** Re-ingest with `--force` to apply them to existing content.
- The "Local" column shows the value in the team's local `.env` as of 13 Oct 2026. A dash means the default is used.

| # | Section |
|---|---|
| 1 | [Question workflow](#1-question-workflow) |
| 2 | [Question flags, step by step](#2-question-flags-step-by-step) |
| 3 | [Ingestion workflow and flags](#3-ingestion-workflow-and-flags) |
| 4 | [Ready-made profiles](#4-ready-made-profiles) |
| 5 | [Checking which flags were applied](#5-checking-which-flags-were-applied) |
| 6 | [Testing a flag change](#6-testing-a-flag-change) |
| 7 | [Pitfalls](#7-pitfalls) |

---

## 1. Question workflow

Each box is a step of `POST /api/v1/chat`. The flag on the right decides whether the step runs, or how. Steps marked **LLM** add a model call, which means more latency and tokens.

```text
 question
    │
    ▼
 ① input guardrail ............................ CONTENT_SAFETY_MODE   off | heuristic
    │  blocked ─► answer_type "blocked"
    ▼
 ② load history ............................... CONVERSATION_STORE    memory | mongo
    ▼
 ③ understand the question  (LLM) ............. QUERY_ANALYSIS_ENABLED, ASSISTANT_SCOPE
    │  intent = chitchat     ─► small-talk reply, no retrieval
    │  intent = out_of_scope ─► canned "outside what I can help with", no retrieval
    ▼
 ④ answer cache ............................... ANSWER_CACHE_ENABLED
    │  hit ─► cached answer (cached: true)
    ▼
 ⑤ choose the mode ............................ RAG_MODE, RAG_ENABLED_MODES, RAG_ALLOW_MODE_OVERRIDE
    ▼
 ⑥ retrieve
    │   semantic: transform (LLM?) → hybrid search → fuse → threshold → recency → MMR → expand
    │   graph:    entity linking → graph search → PageRank → semantic fallback
    │   agentic:  plan → tools → grade → refine (LLM, several calls)
    ▼
 ⑦ flag injected passages ..................... CONTENT_SAFETY_MODE
    ▼
 ⑧ assemble context ........................... RERANKER, RETRIEVAL_KNEE_CUTOFF, CONTEXT_*
    │   dedupe → rerank (LLM?) → score floor → knee cutoff → token budget → reorder
    ▼
 ⑨ generate  (LLM) ............................ PROMPT_VERSIONS, GENERATION_*
    ▼
 ⑩ output guardrail ........................... BLOCK_OUTPUT_PII, CONTENT_SAFETY_MODE
    ▼
 ⑪ assess quality  (LLM if judged) ............ QUALITY_JUDGE, QUALITY_JUDGE_SAMPLE_RATE
    │  confidence low ─► ⑫ escalate ........... RAG_ESCALATION_ENABLED (repeats ⑥–⑪ in a stronger mode)
    ▼
 ⑬ save turn, cache answer, audit, metrics ... LOG_QUESTION_TEXT, RAG_ALLOW_TRACE
```

The code for each step is listed in [workflow-execution.md, section 3](workflow-execution.md#3-ask-questions-what-happens-at-runtime).

---

## 2. Question flags, step by step

### ① Input guardrail

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `CONTENT_SAFETY_MODE` | `heuristic` | `heuristic` | `heuristic`: pattern checks for prompt injection and leaked PII; no cost. There is no harm-category check. | `off`: no checks on questions, passages or answers |

### ③ Understanding the question

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `QUERY_ANALYSIS_ENABLED` | `true` | — | One utility-model call: rewrites follow-ups, fixes spelling, detects intent and complexity, extracts filters ("policies from 2023") | Pattern rules only: greetings become chitchat, everything else is searched; follow-ups aren't rewritten; no filters |
| `ASSISTANT_SCOPE` | blank | *set* | One-line description of what the assistant covers. The intent check uses it to decide what's out of scope. | Blank: the model only knows it's an "enterprise knowledge assistant" and may reject valid but unusual questions (see [pitfalls](#7-pitfalls)) |
| `ORGANISATION_NAME` | `the organisation` | — | Used in prompts | — |

### ④ Answer cache

| Flag | Default | Local | On | Off |
|---|---|---|---|---|
| `ANSWER_CACHE_ENABLED` | `false` | `false` | A similar earlier question with the same permissions returns the stored answer. Similarity is set by `ANSWER_CACHE_SIMILARITY` (0.95), lifetime by `ANSWER_CACHE_TTL_SECONDS` (3600). | Every question is answered fresh |

Keep the cache **off** while testing flags, or you'll see old answers. The playground (`/admin/rag/evaluate`) never uses the cache.

### ⑤ Choosing the mode

| Flag | Default | Local | Effect |
|---|---|---|---|
| `RAG_MODE` | `semantic` | `hybrid` | Mode used when the request doesn't name one: `semantic` (vector only), `hybrid` (vector + BM25), `hybrid_graph` (hybrid + knowledge graph), `graph`, `agentic`. `auto` picks by question complexity, using `RAG_ROUTER_MAP`. |
| `RAG_ENABLED_MODES` | `semantic` | `hybrid,agentic` | Modes this deployment builds and accepts. A request for any other mode gets a `400`. `hybrid_graph` and `graph` need the graph database. |
| `RAG_ALLOW_MODE_OVERRIDE` | `true` | `true` | `true`: a request may set `rag_mode` and `vector_store`. `false`: always `RAG_MODE`. Admin playground runs may always choose. |
| `RAG_ROUTER_MAP` | simple→hybrid, relationship→graph, global→graph, multi_step→agentic | — | Only for `RAG_MODE=auto` |

### ⑥ Retrieval: semantic mode

These run in this order inside `rag/strategies/semantic.py`.

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `VECTOR_STORE_PROVIDER` / `VECTOR_STORE_PROVIDERS` | `mongo` / blank | `mongo` / blank | Read store, and the stores ingestion keeps in sync (only `mongo` today) | — |
| `RETRIEVAL_MODE` | `hybrid` | `semantic_hybrid` | `vector`: embeddings only. `hybrid`: embeddings + BM25 keywords. `semantic_hybrid`: hybrid + an app-side reranker (`MONGODB_SEMANTIC_RERANKER`: `llm`, `cohere` or `none`). | — |
| `MONGODB_TEXT_ANALYZER` | `lucene.english` | — | Analyzer of the Atlas Search (BM25) index. A change needs the text index rebuilt (drop it; it is recreated on startup). | — |
| `SEMANTIC_QUERY_TRANSFORM` | `none` | — | **LLM.** `multi_query`: N rephrasings (`SEMANTIC_MULTI_QUERY_COUNT`). `hyde`: searches with a hypothetical answer. `step_back`: also searches a broader question. | `none`: the question as written |
| `RETRIEVAL_TOP_K` | `8` | — | Passages fetched per search | — |
| `RETRIEVAL_MIN_SCORE` | `0.0` | — | Drops results below this score | `0`: keeps all |
| `SEMANTIC_RECENCY_BOOST` | `0.0` | — | e.g. `0.3`: newest documents score up to 30% higher (half-life `SEMANTIC_RECENCY_HALF_LIFE_DAYS`) | `0`: dates ignored |
| `SEMANTIC_MMR` | `false` | — | Diversifies results so near-duplicates don't fill the top-k (`SEMANTIC_MMR_LAMBDA`) | Pure relevance order |
| `SEMANTIC_EXPAND` | `none` | `parent` | `parent`: replaces a matched chunk with its parent section (works with `CHUNK_PARENT_SIZE` > 0). `window`: adds the neighbouring chunks. | Chunks only |
| `SEMANTIC_HIERARCHICAL` | `false` | — | Finds the best documents first via their summaries, then chunks inside them. Needs `ENRICH_PROFILE=true` at ingestion. | Flat chunk search |

### ⑥ Retrieval: graph and agentic modes

| Flag | Default | Local | Effect |
|---|---|---|---|
| `GRAPH_SEARCH_MODE` | `auto` | — | `local`, `path`, `hybrid`, `global`, `drift`, `dual`, `cypher`, or `auto` to pick per question |
| `GRAPH_USE_PPR` | `true` | — | Ranks graph facts with personalised PageRank |
| `GRAPH_KNEE_CUTOFF` | `false` | `true` | Drops low-scoring facts and community reports after the sharp drop in scores |
| `GRAPH_SEMANTIC_FALLBACK` | `true` | — | Falls back to semantic search when no entities match |
| `GRAPH_ALLOW_FREEFORM_CYPHER` | `false` | — | Lets the LLM write Cypher. Only safe without per-user ACLs. |
| `AGENTIC_TOOLS` | vector, keyword, fetch_document, graph, graph_global, calculator, date_math | — | Tools the agent may call. Graph tools need the graph store; `web` also needs `WEB_SEARCH_*`. |
| `AGENTIC_MAX_ITERATIONS` / `AGENTIC_MAX_TOOL_CALLS` / `AGENTIC_MAX_TOKENS` / `AGENTIC_MAX_SECONDS` | 3 / 12 / 60000 / 90 | — | Hard limits per question |
| `AGENTIC_GENERATE_AND_CHECK` | `true` | — | **LLM.** The agent checks its answer is grounded and complete, and revises it if not |
| `AGENTIC_ALLOW_CLARIFY` | `false` | — | Lets the agent ask the user a clarifying question instead of answering |

### ⑧ Context assembly (all modes)

Order: dedupe → rerank → score floor → knee cutoff → token budget → reorder.

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `RERANKER` | `none` | — | `llm`: **LLM** reranking. `cohere`: Cohere rerank (`RERANKER_ENDPOINT`). | `none`: keeps the search score |
| `RETRIEVAL_RERANK_TOP_N` | `4` | `8` | Passages kept after reranking | — |
| `CONTEXT_MIN_RELATIVE_SCORE` | `0.0` | — | Drops passages below this share of the best score | `0`: keeps all |
| `RETRIEVAL_KNEE_CUTOFF` | `false` | `true` | Drops passages after the sharp drop in scores (`RETRIEVAL_KNEE_SENSITIVITY`) | Keeps all top-n |
| `CONTEXT_MAX_TOKENS` | `6000` | — | Token budget for the passages sent to the model | — |
| `CONTEXT_REORDER` | `true` | — | Puts the strongest passages at the start and end ("lost in the middle") | Score order |

### ⑨–⑪ Generation, output check and quality

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `API_GATEWAY_URL` | blank | *set* | Every LLM and embedding call goes to `<this>/api/llm`, carrying the caller's bearer token. The LLM gateway picks the model for answers and for question analysis, transforms, reranking, grading and the judge, and handles failover. | Blank (and no `LLM_GATEWAY_URL`): readiness fails, no LLM calls |
| `LLM_GATEWAY_URL` | blank | — | Direct llm-gateway URL; overrides `API_GATEWAY_URL` + `/api/llm` | — |
| `LLM_GATEWAY_API_KEY` | blank | — | Sent as `X-API-Key` when there is no caller token: CLI ingestion, scripts, `AUTH_MODE=none` | Blank: those calls are refused by the gateway |
| `LLM_CALLER` / `LLM_TIMEOUT_SECONDS` | `knowledge-service` / `120` | — | Caller name in the gateway's per-caller stats (suffixed with the call purpose), and the per-call timeout | — |
| `GENERATION_TEMPERATURE` / `GENERATION_MAX_TOKENS` | `0.1` / `1024` | — | Sampling temperature and answer length sent to the gateway | — |
| `PROMPT_VERSIONS` | `{}` | — | e.g. `{"rag_answer": "v2"}` picks a prompt version from `src/app/prompts/` | Default versions |
| `BLOCK_OUTPUT_PII` | `true` | — | Withholds answers containing PII that appears in neither the question nor the passages | Answers are returned as generated |
| `QUALITY_JUDGE` | `request` | `request` | **LLM.** `request`: judges when the request sets `"evaluate": true`. `always`: every answer. | `off`: confidence from citation signals only |
| `QUALITY_JUDGE_SAMPLE_RATE` | `0.0` | `0.0` | Share of other answers judged in the background, for dashboards only | — |
| `RAG_ESCALATION_ENABLED` | `false` | `false` | A low-confidence answer is retried in the next stronger enabled mode (up to `RAG_MAX_ESCALATIONS`) | One attempt |

How the judge scores groundedness and how confidence is decided: [README, Answer quality](../README.md#answer-quality).

### ⑬ Logging and debugging

| Flag | Default | Local | Effect |
|---|---|---|---|
| `RAG_ALLOW_TRACE` | `true` | `true` | Allows `"include_trace": true` to return every step with timings. Needed by the playground's trace view. |
| `LOG_QUESTION_TEXT` | `false` | — | Logs the question text with each answer. May contain PII. |
| `AUTH_MODE` | `none` | `none` | `none` is local only: no sign-in, **ACLs not enforced**. QA and Prod use `gateway`: identity comes from the `X-User-ID`, `X-User-Email`, `X-User-Name`, `X-Account-ID` and `X-Is-Admin` headers the API gateway injects, and a request without `X-User-ID` gets `401`. |
| `ACL_ENFORCEMENT` | `true` | — | Filters every store query by the user's permissions: `*`, the user ID and the account ID |
| `ADMIN_REQUIRES_ADMIN` / `INGESTION_REQUIRES_ADMIN` | `true` / `false` | — | In `gateway` mode, the admin RAG studio and the ingestion/deletion endpoints need `X-Is-Admin: true` (`403` otherwise) |

---

## 3. Ingestion workflow and flags

```text
 file / source
    │
    ▼
 raw checks ......................... INGESTION_MAX_FILE_MB, RETENTION_MAX_AGE_DAYS
    ▼
 land raw copy ...................... ZONES_STORE              off | local (.ragdata/)
    ▼
 parse .............................. PDF_PARSER               native (pypdf, text only)
    ▼
 clean & govern ..................... PII_MODE, INGESTION_INJECTION_ACTION, INGESTION_MAX_SENSITIVITY
    │  fails a check ─► quarantine (reason in the manifest)
    ▼
 change detection ................... MANIFEST_STORE           unchanged ─► skip (use --force to redo)
    ▼
 chunk .............................. CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_PARENT_SIZE
    ▼
 enrich  (LLM) ...................... ENRICH_PROFILE, ENRICH_CONTEXTUAL, ENRICH_QUESTIONS, ENRICH_SUMMARIES
    ▼
 embed .............................. EMBEDDING_CACHE_ENABLED, EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE
    ▼
 write .............................. INGESTION_TARGETS        vector | graph | vector,graph
                                      VECTOR_STORE_PROVIDERS   mongo
```

| Flag | Default | Local | On / value | Off |
|---|---|---|---|---|
| `INGESTION_TARGETS` | `vector` | `vector` | `graph` adds **LLM** entity and relation extraction into the MongoDB graph collections | Vector index only |
| `MANIFEST_STORE` | `memory` | `memory` | `mongo`: document status survives restarts | `memory`: **the document list is wiped on every restart**, so `GET /ingestion/manifest` comes back empty and unchanged files are re-processed |
| `ZONES_STORE` | `local` | `local` | Keeps raw, parsed, clean and chunk copies under `ZONES_LOCAL_ROOT` (`.ragdata/`) for debugging | `off`: nothing kept |
| `PDF_PARSER` | `native` | — | `native` is the only value: the pypdf text layer. Pictures inside PDFs are ignored and image-only (scanned) pages give no text. Image and audio/video files are not ingested. | — |
| `PII_MODE` | `off` | `regex` | `regex` finds pattern-shaped PII such as e-mail addresses and phone numbers (not names or postal addresses); `PII_ACTION` masks or flags it | No PII scan |
| `INGESTION_INJECTION_ACTION` | `quarantine` | — | Documents containing prompt injection are quarantined | `flag`: indexed with a warning |
| `CHUNK_STRATEGY` | `structure` | `semantic` | `structure`: splits on headings and paragraphs. `semantic`: splits where the topic changes. | — |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | `512` / `64` | Tokens per chunk, and tokens repeated between chunks | — |
| `CHUNK_PARENT_SIZE` | `0` | `1500` | > 0: small chunks are searched, and their larger parent section is returned (pair with `SEMANTIC_EXPAND=parent`) | `0`: single-level chunks |
| `ENRICH_PROFILE` | `false` | — | **LLM.** Document summary, type and tags. Needed by `SEMANTIC_HIERARCHICAL`. | — |
| `ENRICH_CONTEXTUAL` | `false` | `true` | **LLM.** Adds a sentence of document context to each chunk before embedding | — |
| `ENRICH_QUESTIONS` | `0` | `3` | **LLM.** Generates N questions per chunk and indexes them | `0`: none |
| `ENRICH_SUMMARIES` | `false` | — | **LLM.** A summary per chunk, indexed alongside it | — |
| `EMBEDDING_CACHE_ENABLED` | `true` | — | Re-ingesting unchanged text doesn't pay for embeddings again | — |

**After changing any chunking, enrichment or parser flag**, re-ingest with `--force`, otherwise unchanged documents are skipped and keep their old chunks:

```bash
python scripts/ingest_cli.py --source sample/ --collection default --force
```

Changing `EMBEDDING_DIMENSIONS`, the LLM gateway's embedding model or the index schema needs a [full re-index](runbooks/reindex.md). `EMBEDDING_DIMENSIONS` must match the gateway's embedding model.

---

## 4. Ready-made profiles

Copy a block into `.env` and restart. Flags not listed keep their current value.

**Cheapest: semantic only, fewest LLM calls**
```ini
RAG_ENABLED_MODES=semantic
RAG_MODE=semantic
RETRIEVAL_MODE=hybrid
SEMANTIC_QUERY_TRANSFORM=none
RERANKER=none
QUALITY_JUDGE=off
ENRICH_CONTEXTUAL=false
ENRICH_QUESTIONS=0
```
One LLM call per question (question analysis is cheap; set `QUERY_ANALYSIS_ENABLED=false` to drop it too, at the cost of follow-up rewriting).

**Best retrieval quality (current local setup, roughly)**
```ini
RETRIEVAL_MODE=semantic_hybrid
CHUNK_PARENT_SIZE=1500
SEMANTIC_EXPAND=parent
RETRIEVAL_KNEE_CUTOFF=true
RETRIEVAL_RERANK_TOP_N=8
ENRICH_CONTEXTUAL=true
ENRICH_QUESTIONS=3
```
Re-ingest with `--force` after switching to it.

**Evaluation and flag testing**
```ini
RAG_ALLOW_MODE_OVERRIDE=true
RAG_ALLOW_TRACE=true
ANSWER_CACHE_ENABLED=false
QUALITY_JUDGE=request
RAG_ESCALATION_ENABLED=false
ASSISTANT_SCOPE=Questions about any documents uploaded to the knowledge base, including test and sample documents
MANIFEST_STORE=mongo
```
Escalation is off so each run measures one mode only. `MANIFEST_STORE=mongo` needs `MONGODB_URL`.

**Production (starting point)**
```ini
AUTH_MODE=gateway
API_GATEWAY_URL=<API gateway URL>
RAG_ALLOW_MODE_OVERRIDE=false
RAG_ALLOW_TRACE=false
QUALITY_JUDGE=request
QUALITY_JUDGE_SAMPLE_RATE=0.05
CONTENT_SAFETY_MODE=heuristic
MANIFEST_STORE=mongo
CONVERSATION_STORE=mongo
ASSISTANT_SCOPE=<what the assistant actually covers>
```
Set these as the service's environment variables. `AUTH_MODE=gateway` is only safe when the service can be reached through the API gateway alone.

---

## 5. Checking which flags were applied

| What | How |
|---|---|
| Modes enabled, and missing settings | `GET /api/v1/health/ready` lists configuration errors. The startup log line `app_startup` shows `rag_mode` and `enabled_modes`. |
| Retrieval settings and models in play | `GET /api/v1/admin/rag/config` (RAG Studio shows it) |
| What a single question did | Add `"include_trace": true` to a chat request, or open the trace view in the playground. Each step shows its timing and details: `input_guardrail`, `query_analysis`, `router` (mode and complexity), `cache_lookup`, `query_transform`, `vector_search`, `expand`, `assemble_context` (counts after rerank and knee), `generate`, `output_guardrail`, `escalation`. |
| How the question was understood | `understanding` in the response: standalone question, `intent`, `complexity`, filters, and `analysed_by` (`llm` or `heuristic`) |

---

## 6. Testing a flag change

1. **Record a baseline** with the current flags:
   ```bash
   python scripts/run_eval.py --rag-mode all --judge --save-baseline
   ```
2. **Change one flag** in `.env`, restart the API, and re-ingest with `--force` if it's an ingestion flag.
3. **Run the golden set again** and compare:
   ```bash
   python scripts/run_eval.py --rag-mode all --judge --gate
   ```
   The table shows hit, recall, precision, MRR, nDCG, citation precision and recall, groundedness (`faithfulness`), correctness, latency and tokens per mode. `--gate` fails on scores below `evals/thresholds.yaml` or regressions against the baseline.
4. **To test many variants at once**, list them in `evals/ablations.yaml` and run `python scripts/run_ablation.py`.

For a single question, use the RAG Studio **Playground**. Pick a golden-set question so the expected sources and reference answer are filled in, then compare runs in the **Compare** tab.

**Sample test document.** `data/sample/toy-room-inventory.pdf` is a 2-page PDF with simple facts (8 blue and 10 red balls in Bag A, and so on). The questions `toy-q1` to `toy-q11` in the golden set cover lookups, arithmetic, two-page facts, a relationship question and an unanswerable one. Rebuild the PDF with `python scripts/make_sample_pdf.py`.

---

## 7. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| "That question is outside what I can help with" for a question your documents answer | `ASSISTANT_SCOPE` is blank or too narrow, so question analysis marks it `out_of_scope` and retrieval never runs | Set `ASSISTANT_SCOPE` to cover the content, and restart. Check `understanding.intent` in the response. |
| Every retrieval metric is 0% in the playground, although the answer is right | The **Expected sources** value isn't the end of the stored source path. An uploaded file is stored as `upload://<name>`, so `sample/<name>` never matches. The field also keeps its value when you type a new question. | Use just the file name, e.g. `toy-room-inventory.pdf` |
| No retrieval tiles at all in the playground | Expected sources is empty, so there's nothing to score against | Fill it in, or load a golden-set question |
| Recall@k or nDCG above 100% | Known bug: several chunks from the same expected source are each counted as a hit | Read recall as "≥ 1 means found" until fixed |
| The manifest is empty after a restart | `MANIFEST_STORE=memory` | Use `mongo`, or re-check documents with `/retrieval` |
| A flag change made no difference | The API wasn't restarted; or it's an ingestion flag and documents weren't re-ingested with `--force`; or the answer came from the cache | Restart, re-ingest with `--force`, set `ANSWER_CACHE_ENABLED=false` |
| `400` "RAG mode … is not enabled" | The mode isn't in `RAG_ENABLED_MODES`, or `RAG_ALLOW_MODE_OVERRIDE=false` | Add the mode, or drop `rag_mode` from the request |
| Questions about a diagram or scanned page get "not found" | PDFs are parsed as text only: pictures inside them are ignored, and image-only pages give no text (`pages_without_text` quality flag) | Provide the content as text (a text-based PDF, or a document that describes the diagram) and re-ingest it. See [workflow-execution.md, 2.2.1](workflow-execution.md#221-images-and-diagrams). |
| Graph questions get semantic answers | Graph mode isn't enabled, or `INGESTION_TARGETS` doesn't include `graph` so the graph is empty and `GRAPH_SEMANTIC_FALLBACK` kicks in | Enable graph mode, ingest with `--targets graph`, and run graph maintenance |

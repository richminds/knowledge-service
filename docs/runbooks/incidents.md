# Runbook: incidents (C-243)

On-call: TODO (name the distribution list).
Severity: **Sev1** = data leak or full outage · **Sev2** = degraded answers or a partial outage · **Sev3** = cost or quality drift.

## Outage: 5xx errors or readiness failing

1. Call `GET /api/v1/health/ready`. It lists each dependency and any configuration errors.
2. Search the service's JSON logs (stdout) for `"level": "error"` lines and the `unhandled_exception` event. Group by `path` and exception type.
3. Check for these common causes:
   - LLM gateway errors (`LLM call '<purpose>' failed: …`, with the gateway's status and code). A 429 means the user's or the service's limit or budget there is exhausted; a 401 or 403 on CLI or script calls means `LLM_GATEWAY_API_KEY` is missing or wrong. Model failover is the gateway's job, so check its health and logs too.
   - MongoDB Atlas throttling or connection limits (Atlas metrics). Scale the cluster tier.
   - Atlas network access: the app's outbound IPs must be in the project's access list.
4. If the problem started with a release, [roll back](rollback.md).

## Bad or harmful answers

1. Find the response. Users give its `response_id`, which also appears in feedback. In the JSON logs, filter on `request_id` equal to the `response_id`.
2. Classify the failing stage: retrieval missed it, a wrong document was retrieved, generation ignored the context, or a stale document was retrieved.
3. Mitigate:
   - Remove or fix the document with `DELETE /api/v1/ingestion/sources?source=...`.
   - Roll back the prompt version with `PROMPT_VERSIONS`.
   - Switch the RAG mode.
4. Add the question to `evals/golden_set.jsonl` (C-238).

## Data leak (Sev1)

The user saw content they should not see, or PII was exposed.

1. **Contain.** Delete the source (`DELETE /api/v1/ingestion/sources`). This removes it from the vector index, the graph and the manifest.
2. Clear the answer cache. Restart the app, or flush Redis if the cache is enabled.
3. **Scope.** In the JSON logs, find the `rag_answer` and `audit` events that mention the source.
4. **Root cause.** Find which of these went wrong:
   - The ACL was wrong at the source.
   - The ACL mapping is wrong (`config/acl_mapping.yaml`).
   - `"*"` was applied by `INGESTION_DEFAULT_ACL`.
   - A retriever has a bug. Add a failing test to `tests/integration/test_isolation_redteam.py`.
5. Notify privacy/compliance according to policy (TODO: contact). Handle any user deletion requests with `DELETE /api/v1/conversations/me`, run by the user, or by an admin script.

## Cost spike

1. Check the LLM gateway's per-caller stats: calls are labelled `LLM_CALLER:<purpose>`. Look for a runaway agent (`agentic_*`), the judge sample rate, or a re-ingest.
2. Tighten the limits:
   - `AGENTIC_MAX_TOKENS` and `AGENTIC_MAX_TOOL_CALLS`
   - `QUALITY_JUDGE_SAMPLE_RATE`
   - `RATE_LIMIT_PER_MINUTE`

## Failed ingestion

1. Check `GET /api/v1/ingestion/manifest?status=failed`, the API job status (`GET /api/v1/ingestion/{job_id}`), and the scheduler's history for `scripts/reconcile.py`.
2. Fix the cause, then re-run the ingest or `scripts/reconcile.py`.

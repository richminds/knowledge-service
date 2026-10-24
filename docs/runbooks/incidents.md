# Runbook: incidents (C-243)

On-call: TODO (the distribution list is `alertEmails` in `infra/*.bicepparam`).
Severity: **Sev1** = data leak or full outage · **Sev2** = degraded answers or a partial outage · **Sev3** = cost or quality drift.

## Outage: 5xx errors or readiness failing

1. Call `GET /api/v1/health/ready`. It lists each dependency and any configuration errors.
2. App Insights → Failures. Group by `operation_Name` and exception type.
3. Check for these common causes:
   - Azure OpenAI 429 errors. Check the quota, and set the fallback with `AZURE_OPENAI_FALLBACK_ENDPOINT`.
   - MongoDB Atlas throttling or connection limits (Atlas metrics). Scale the cluster tier.
   - Atlas network access: the app's outbound IPs must be in the project's access list.
4. If the problem started with a release, [roll back](rollback.md).

## Bad or harmful answers

1. Find the response. Users give its `response_id`, which also appears in feedback. In Log Analytics:
   ```kusto
   traces | where customDimensions.request_id == "<response_id>"
   ```
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
3. **Scope.** Audit log query:
   ```kusto
   traces | where message in ("rag_answer", "audit") | where customDimensions has "<source>"
   ```
4. **Root cause.** Find which of these went wrong:
   - The ACL was wrong at the source.
   - The ACL mapping is wrong (`config/acl_mapping.yaml`).
   - `"*"` was applied by `INGESTION_DEFAULT_ACL`.
   - A retriever has a bug. Add a failing test to `tests/integration/test_isolation_redteam.py`.
5. Notify privacy/compliance according to policy (TODO: contact). Handle any user deletion requests with `DELETE /api/v1/conversations/me`, run by the user, or by an admin script.

## Cost spike

1. Open the workbook's **Tokens by purpose** panel. Look for a runaway agent (`agentic_*`), the judge sample rate, or a re-ingest.
2. Tighten the limits:
   - `AGENTIC_MAX_TOKENS` and `AGENTIC_MAX_TOOL_CALLS`
   - `QUALITY_JUDGE_SAMPLE_RATE`
   - `RATE_LIMIT_PER_MINUTE`

## Ingestion backlog or dead letters

1. Check the Service Bus queue `ingestion` in the portal: count the active and dead-letter messages.
2. Read the reasons on the dead-letter messages, and check `GET /api/v1/ingestion/manifest?status=failed`.
3. Fix the cause, then resubmit the messages from the dead-letter queue (with Service Bus Explorer), or run `scripts/reconcile.py`.

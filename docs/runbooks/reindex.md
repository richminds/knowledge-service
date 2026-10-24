# Runbook: full re-index (C-152, C-138)

Use this runbook when the **embedding model**, the **index schema**, the **chunking settings** or the **parsers** change. Everyday changes are handled automatically by incremental sync and do not need it.

The app reads one chunk collection (`MONGODB_CHUNKS_COLLECTION`) with two Atlas Search indexes (`MONGODB_VECTOR_INDEX`, `MONGODB_SEARCH_INDEX`). A new collection is built next to the live one, evaluated, and then switched in by changing those settings. This is the template's "alias". Users see no downtime.

## Steps

1. **Pick new names.** For example, collection `rag_chunks_v2` with indexes `rag_chunks_v2_vector` and `rag_chunks_v2_text`.
2. **Build it from a workstation or a one-off job**, with the new settings:
   ```bash
   export MONGODB_CHUNKS_COLLECTION=rag_chunks_v2
   export MONGODB_VECTOR_INDEX=rag_chunks_v2_vector MONGODB_SEARCH_INDEX=rag_chunks_v2_text
   export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<new model>      # if the model changes
   export EMBEDDING_DIMENSIONS=<dims>
   python scripts/create_index.py
   python scripts/ingest_cli.py --source <each source> --collection <c> --force --full-sync
   ```
   `--force` re-processes unchanged documents. `--full-sync` removes documents that are gone from the source.
3. **Reconcile counts (C-139).** `python scripts/reconcile.py` prints indexed counts against the manifest. Investigate any mismatch, starting with the documents in quarantine (`GET /api/v1/ingestion/manifest?status=quarantined`).
4. **Evaluate** against the golden set with the new index:
   `python scripts/run_eval.py --rag-mode all --judge --gate`. Proceed only if the gate passes.
5. **Graph (if enabled).** A parser or chunking change also changes chunk IDs. Run `python scripts/graph_maintenance.py all` after the ingest.
6. **Switch.** Set the three `MONGODB_*` names in the App Service settings (or in Bicep) on the **staging slot** and smoke-test it, then swap. Do the same for the jobs.
7. **Clean up.** Keep the old collection for 7 days so you can roll back, then drop it, which also removes its search indexes.

## Adding a second vector store

Once another provider is implemented in `rag/vectorstore/`, set `VECTOR_STORE_PROVIDERS=mongo,<other>` (the default for reads stays `VECTOR_STORE_PROVIDER`). From then on every ingest writes to both stores, and `vector_store` on a request (the RAG Studio "Vector DB" toggle) picks which one answers. Documents ingested earlier are only in the first store and are skipped as unchanged, so back-fill the new one once:

```bash
python scripts/ingest_cli.py --source <each source> --collection <c> --force
```

`--force` re-processes every document (embeddings are reused when `EMBEDDING_CACHE_ENABLED` is on). Reconciliation (C-139) uses the smaller of the two counts, so a store that fell behind is reported. Both stores must use the same `EMBEDDING_DIMENSIONS` (Atlas Vector Search supports up to 8192).

## Rollback

Set the `MONGODB_*` names back to the old collection and indexes, or swap the slots back. The old collection is untouched.

## Cost note

A full re-embed costs about `chunks × 500` embedding tokens. Graph re-extraction costs about `chunks × 2,000` utility-model tokens, though cached chunks are skipped.

# Runbook: roll back a release (C-235)

## Any environment

Redeploy the previous image. Tag every build you deploy (for example `oissa-rag-agent:<git sha>`), so the last good one is always at hand.

1. Stop the current container, or point the service at the previous tag.
2. Start the previous image with the same environment variables.
3. Check `GET /api/v1/health/ready` lists no configuration errors.

If no previous image was kept, check out the previous release commit, build it (`docker build -t oissa-rag-agent:<sha> .`) and run that.

## What redeploying does *not* roll back

| Change | How to roll back |
|---|---|
| Environment variables | Set them back, then restart the service |
| Chunk collection contents/schema | [reindex runbook](reindex.md): point `MONGODB_CHUNKS_COLLECTION` back |
| Prompt version | Set `PROMPT_VERSIONS` back, e.g. `{"rag_answer": "v1"}`. This needs no redeploy, only a restart. |
| RAG mode | Set `RAG_MODE` / `RAG_ENABLED_MODES` back, then restart |
| Scheduled scripts | Point the scheduler at the previous image or commit |
| The LLM gateway's model choice | Roll it back in the gateway's configuration. If its embedding model changed, `EMBEDDING_DIMENSIONS` and the index must match it ([reindex](reindex.md)). |

After a rollback, record the incident ([incidents.md](incidents.md)) and add the failing questions to the golden set.

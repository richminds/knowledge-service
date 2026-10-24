# Runbook: roll back a release (C-235)

## Production (staging slot)

Each prod release is deployed to the `staging` slot, smoke-tested, then **swapped**. After the swap, the previous release is in `staging`.

**Swap back.** This takes one step, and the app does not restart cold:

```bash
az webapp deployment slot swap -g rg-oissa-rag-prod -n <app-name> --slot staging --target-slot production
```

Or in the portal: App Service → Deployment slots → Swap.

## Older release, or QA

Redeploy a previous image tag. Every build pushes `oissa-rag-agent:<BuildId>`.

```bash
az webapp config container set -g <rg> -n <app-name> \
  --container-image-name oissaragagentacr.azurecr.io/oissa-rag-agent:<previous BuildId>
```

Alternatively, re-run the earlier pipeline run's **DeployProd** stage.

## What a swap does *not* roll back

| Change | How to roll back |
|---|---|
| Infrastructure (Bicep) | Redeploy the previous commit's Bicep; most changes are additive |
| Chunk collection contents/schema | [reindex runbook](reindex.md): point `MONGODB_CHUNKS_COLLECTION` back |
| Prompt version | Set `PROMPT_VERSIONS` back, e.g. `{"rag_answer": "v1"}`. This needs no redeploy, only a restart. |
| RAG mode | Set the `ragMode` / `ragEnabledModes` parameters back, then redeploy |
| Ingestion jobs | Update the job image: `az containerapp job update --image ...` |

After a rollback, record the incident ([incidents.md](incidents.md)) and add the failing questions to the golden set.

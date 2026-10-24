# 0006. App Service for the API, Container Apps jobs for workers (C-035, C-045)

Superseded by [0008](0008-llm-gateway-and-gateway-identity.md): the repository no longer provisions or deploys infrastructure.

**Context.**
- The query API is a long-running HTTP service. It needs slots for blue-green releases and simple autoscale.
- Ingestion is bursty and queue-driven.
- Graph maintenance and reconciliation are nightly batches.

**Decision.**
- The API runs on App Service, as a Linux container with a prod staging slot and CPU autoscale.
- The workers are Container Apps jobs that use the same image:
  - an event-triggered job, which a KEDA Service Bus scaler starts based on queue length;
  - cron jobs for the reconciler and graph maintenance.

**Consequences.**
- One image serves both the API and the workers, and the pipeline builds it once.
- AKS was not chosen, because the team would have to operate the cluster.
- Azure Functions was not chosen, because of execution-time limits on long documents.

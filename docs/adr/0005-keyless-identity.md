# 0005. Keyless access with one workload identity (C-046, C-047)

**Context.** Keys in app settings leak, need rotation and cannot be scoped.

**Decision.**
- One user-assigned managed identity is shared by the web app, its staging slot and the jobs.
- It is granted data-plane roles only: Search Index Data Contributor, Storage Blob Data Contributor, Key Vault Secrets User, Cognitive Services (OpenAI) User, Service Bus Sender/Receiver, the Cosmos data contributor role, and Postgres Entra admin.
- Local auth is disabled where the service allows it.
- The few unavoidable secrets (the API key, the Redis URL, connector credentials) live in Key Vault. The app reads them at startup.

**Consequences.**
- No secret is in code, pipeline variables or app settings.
- One identity is simpler than one per service. Split it if workers must not read user conversations (Cosmos) or the API must not write to storage.

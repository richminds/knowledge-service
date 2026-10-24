# Architecture Decision Records (C-017)

Each record covers one decision: its context, the decision, and its consequences. To change a decision, add a new record that supersedes the old one. Do not edit accepted records.

| # | Decision | Status |
|---|---|---|
| [0001](0001-framework-fastapi-langgraph.md) | FastAPI and LangGraph, with no all-in-one RAG framework | Accepted |
| [0002](0002-azure-ai-search-vector-store.md) | Azure AI Search for vector, BM25 and semantic ranking | Superseded by 0007 |
| [0003](0003-postgres-age-graph.md) | PostgreSQL + Apache AGE for the knowledge graph | Superseded by 0007 |
| [0004](0004-rag-mode-switch.md) | One codebase, three RAG modes behind a flag and a router | Accepted |
| [0005](0005-keyless-identity.md) | Keyless access with one workload identity | Accepted |
| [0006](0006-hosting.md) | App Service for the API, Container Apps jobs for workers | Accepted |
| [0007](0007-mongodb-atlas-store.md) | MongoDB Atlas for vectors, keyword search, the graph and the manifest | Accepted |

Template: copy `0001` and keep the headings.

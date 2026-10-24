# 0002. Azure AI Search as the vector and keyword store (C-041)

Superseded by [0007](0007-mongodb-atlas-store.md): vectors and keyword search moved to MongoDB Atlas.

**Context.** Semantic RAG needs dense search, BM25, filtering by ACL and metadata, and ideally a reranker. The platform is Azure.

**Decision.**
- Use one Azure AI Search index. It stores chunks, generated questions, summaries, parents and community reports, distinguished by `kind`.
- Search is hybrid (HNSW + BM25 with RRF), with the optional semantic ranker. ACLs are filtered with `acl/any()`.
- Vectors can be compressed (`AZURE_SEARCH_VECTOR_COMPRESSION`).

**Consequences.**
- There is one service to secure, back up and scale, with Entra data-plane RBAC.
- The index schema is versioned in code. A change requires the [reindex runbook](../runbooks/reindex.md).
- Alternatives considered: pgvector (cheaper, but weaker BM25 and no managed ranker) and Cosmos DB vector search.

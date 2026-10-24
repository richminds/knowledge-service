# 0007. MongoDB Atlas for vectors, keyword search, the graph and the manifest

Supersedes [0002](0002-azure-ai-search-vector-store.md) and [0003](0003-postgres-age-graph.md).

The Key Vault and Bicep consequences below are superseded by [0008](0008-llm-gateway-and-gateway-identity.md): `MONGODB_URL` is an environment variable, and conversations are stored here too.

**Context.** The service ran on two data stores: Azure AI Search for vector and BM25 search, and PostgreSQL with Apache AGE and pgvector for the graph, entity embeddings and the ingestion manifest. Each had its own provisioning, identity setup and operations, including a server restart for AGE preloading. One MongoDB Atlas cluster is already in use for makemerich-backend.

**Decision.**
- All retrieval data lives in one MongoDB Atlas database (`MONGODB_URL`, `MONGODB_DATABASE`).
- Chunks are documents in `rag_chunks`. Vector search uses Atlas Vector Search (`$vectorSearch`, cosine, HNSW). Keyword search uses Atlas Search (`$search`, Lucene BM25). Hybrid search fuses the two with RRF in the app. `semantic_hybrid` reranks in the app with `MONGODB_SEMANTIC_RERANKER`.
- The graph is stored as collections (`rag_graph_chunks`, `_entities`, `_relations`, `_claims`, `_sections`, `_documents`). Traversal runs hop by hop with indexed queries. Entity embeddings are stored on the entity documents and searched with `$vectorSearch`.
- The ingestion manifest (`MANIFEST_STORE=mongo`) and the optional agent checkpointer (`AGENTIC_CHECKPOINTER=mongo`) use the same cluster.
- ACL, collection and metadata filters are applied inside every query.

**Consequences.**
- One store to run, back up and secure. The connection string is a secret (Key Vault `mongodb-url`), because Atlas does not support Entra ID workload identities the way Azure data services do.
- Atlas is not provisioned by the Bicep. The cluster, its database user and its network access list are managed in Atlas.
- There is no Cypher engine: `GRAPH_SEARCH_MODE=cypher` and the agent's `graph_query` tool fall back to local graph search. Templates in `config/cypher_templates.yaml` are unused until a Cypher-capable store is added.
- Azure's built-in semantic ranker is gone. `semantic_hybrid` costs an LLM or Cohere rerank call.
- Atlas Search indexes build asynchronously, so a new collection answers searches a few seconds after the first write. The M0 tier allows only three search indexes, which is exactly what the app creates.
- Networks that decrypt TLS (corporate proxies) can break Atlas authentication. Those hosts need an exemption for `*.mongodb.net:27017`.

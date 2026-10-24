# 0003. PostgreSQL + Apache AGE for the knowledge graph (G-05)

**Context.** Graph RAG needs entities, relations, communities and links to source chunks, plus multi-hop traversal and Cypher. Options considered were Neo4j (needs a separate licence and operations), Cosmos DB Gremlin (has no Cypher), and Azure Database for PostgreSQL with the AGE extension.

**Decision.**
- The graph uses PostgreSQL Flexible Server with Apache AGE for Cypher, and pgvector `halfvec` for entity embeddings.
- The same server hosts the ingestion manifest (`ragops` database).
- Community detection (Louvain) and PageRank run in Python (networkx) in the nightly graph job.

**Consequences.**
- The graph is a managed service with point-in-time restore, Entra authentication and HA in prod.
- AGE must be allow-listed and preloaded. Preloading needs a server restart, which the deploy pipeline does when one is pending.
- Free-form text-to-Cypher is disabled under ACL enforcement. Only templates that return ACL columns are used.

from app.core.config import Settings
from app.rag.graph.base import GraphStore

_graph_store_singleton: GraphStore | None = None


def get_graph_store(settings: Settings) -> GraphStore:
    """Return the configured GraphStore implementation (process-wide singleton)."""
    global _graph_store_singleton
    if _graph_store_singleton is None:
        _graph_store_singleton = _build_graph_store(settings)
    return _graph_store_singleton


def _build_graph_store(settings: Settings) -> GraphStore:
    if settings.graph_store_provider == "mongo":
        from app.rag.graph.mongo_store import MongoGraphStore

        return MongoGraphStore(
            url=settings.mongodb_url.get_secret_value(),
            database=settings.mongodb_database,
            prefix=settings.mongodb_graph_prefix,
            entity_vector_index=settings.mongodb_entity_vector_index,
            embedding_dim=settings.embedding_dimensions,
        )

    if settings.graph_store_provider == "neo4j":
        from app.rag.graph.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password.get_secret_value(),
            database=settings.neo4j_database,
            entity_vector_index=settings.neo4j_entity_vector_index,
            embedding_dim=settings.embedding_dimensions,
            query_timeout_seconds=settings.neo4j_query_timeout_seconds,
        )

    raise ValueError(f"Unsupported graph store provider: {settings.graph_store_provider}")

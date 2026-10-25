"""Neo4j (Aura) graph store: configuration and provider selection. No network: the driver is
created lazily, on the first query."""

import pytest

from app.core.config import RagMode, Settings
from app.core.exceptions import GraphStoreError
from app.rag.graph import factory
from app.rag.graph.neo4j_store import Neo4jGraphStore, _access


@pytest.fixture
def graph_settings(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED_MODES", "hybrid_graph")
    monkeypatch.setenv("RAG_MODE", "hybrid_graph")
    monkeypatch.setenv("GRAPH_STORE_PROVIDER", "neo4j")
    monkeypatch.setattr(factory, "_graph_store_singleton", None)


def test_aura_credentials_file_names_configure_the_store(graph_settings, monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://abc123.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    settings = Settings()

    assert settings.graph_enabled and settings.graph_store_configured
    assert settings.configuration_errors() == []
    store = factory.get_graph_store(settings)
    assert isinstance(store, Neo4jGraphStore) and store.supports_cypher


def test_previous_implementation_graph_names_still_work(graph_settings, monkeypatch):
    monkeypatch.setenv("RAG_GRAPH_DB_URI", "neo4j+s://old.databases.neo4j.io")
    monkeypatch.setenv("RAG_GRAPH_DB_USER", "old-user")
    monkeypatch.setenv("RAG_GRAPH_DB_PASSWORD", "old-secret")
    settings = Settings()

    assert settings.neo4j_uri == "neo4j+s://old.databases.neo4j.io"
    assert settings.neo4j_username == "old-user"
    assert settings.graph_store_configured


def test_missing_aura_credentials_are_reported(graph_settings):
    settings = Settings()

    assert RagMode.HYBRID_GRAPH in settings.rag_enabled_modes
    assert "NEO4J_URI and NEO4J_PASSWORD are required for graph RAG" in (
        settings.configuration_errors()
    )
    with pytest.raises(GraphStoreError):
        Neo4jGraphStore(uri="", username="neo4j", password="")


def test_access_clause_is_fail_closed_and_skipped_when_not_enforced():
    assert _access("c", None, None) == "true"
    clause = _access("c", frozenset(), "acct-1")
    assert "$principals" in clause and "$account_id" in clause

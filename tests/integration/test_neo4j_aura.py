"""Neo4j graph store against a real instance (Aura): ingestion writes, retrieval reads, ACL,
Cypher templates and deletes, in a throwaway collection.

Opt-in: reads NEO4J_* from .env / .env.local and runs only with RUN_NEO4J_TESTS=1
    RUN_NEO4J_TESTS=1 pytest tests/integration/test_neo4j_aura.py
"""

import os
import uuid

import pytest

from app.core.config import Settings
from app.rag.graph.base import ChunkGraph, Claim, Entity, Relation
from app.rag.graph.cypher import FACT_COLUMNS, load_templates
from app.rag.graph.neo4j_store import Neo4jGraphStore
from app.rag.types import RetrievalFilters

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1", reason="set RUN_NEO4J_TESTS=1 to run against Neo4j"
)


@pytest.fixture
async def store():
    settings = Settings(_env_file=(".env", ".env.local"))
    if not settings.graph_store_configured or settings.graph_store_provider != "neo4j":
        pytest.skip("GRAPH_STORE_PROVIDER=neo4j and NEO4J_* are not set in .env")
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
        entity_vector_index=settings.neo4j_entity_vector_index,
        embedding_dim=settings.embedding_dimensions,
    )
    graph.test_dimensions = settings.embedding_dimensions
    await graph.ping()
    await graph.ensure_schema()
    yield graph
    await graph.close()


def _chunk(collection: str, chunk_id: str, acl: list[str], account: str, **kwargs) -> ChunkGraph:
    return ChunkGraph(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        collection=collection,
        content=f"content of {chunk_id}",
        acl=acl,
        account_id=account,
        doc_id=chunk_id,
        title=chunk_id,
        section_path=["Intro"],
        **kwargs,
    )


async def test_ingest_retrieve_and_delete(store):
    collection = f"test_{uuid.uuid4().hex[:8]}"
    try:
        await store.upsert_chunk_graph(
            _chunk(
                collection, f"{collection}-c1", ["*"], "acct-1",
                entities=[Entity("Alice", "Person"), Entity("Finance", "Department")],
                relations=[Relation("Alice", "Finance", "MANAGES", "runs it", 0.9)],
                claims=[Claim("Finance", "Budget approved", "2026-01-01")],
            )
        )  # fmt: skip
        await store.upsert_chunk_graph(
            _chunk(
                collection, f"{collection}-c2", ["grp-secret"], "acct-1",
                entities=[Entity("Mallory", "Person"), Entity("Finance", "Department")],
                relations=[Relation("Mallory", "Finance", "MANAGES")],
            )
        )  # fmt: skip
        public = RetrievalFilters(
            collection=collection, principals=frozenset({"*"}), account_id="acct-1"
        )
        other_account = RetrievalFilters(
            collection=collection, principals=frozenset({"*"}), account_id="acct-2"
        )

        found = await store.find_entities(["finance", "mallory"], public, 5)
        assert [e.key for e in found] == ["finance"]  # Mallory: only in an unreadable chunk
        assert await store.find_entities(["finance"], other_account, 5) == []

        facts = await store.neighborhood(["finance"], public, hops=2, limit=10)
        assert [(f.source_name, f.relation, f.target_name) for f in facts] == [
            ("Alice", "MANAGES", "Finance")
        ]
        chunks = await store.chunks_for_entities(["alice", "finance"], public, 5)
        assert [c.chunk_id for c in chunks] == [f"{collection}-c1"]
        assert chunks[0].score == 1.0
        claims = await store.claims_for_entities(["finance"], public, 5)
        assert [(c.text, chunk) for c, chunk in claims] == [("Budget approved", f"{collection}-c1")]
        mentioned = await store.entities_for_chunks([f"{collection}-c1"], public)
        assert {e.key for e in mentioned} == {"alice", "finance"}

        template = next(t for t in load_templates("config/cypher_templates.yaml")
                        if t.name == "who_manages")  # fmt: skip
        rows = await store.run_cypher(
            template.query, {"entity": "finance", "collection": collection}, FACT_COLUMNS
        )
        assert {(r[0], r[6]) for r in rows} == {("Alice", "acct-1"), ("Mallory", "acct-1")}

        vector = [1.0] + [0.0] * (store.test_dimensions - 1)
        await store.upsert_entity_embeddings(collection, [("alice", "Alice", vector)])
        await store.set_entity_properties(collection, {"alice": {"rank": 0.5, "bogus": 1}})
        export = await store.export_graph(collection, frozenset({"*"}))
        assert set(export.nodes) == {"alice", "finance"} and len(export.edges) == 1
        assert export.nodes["alice"].rank == 0.5

        assert await store.update_acl(f"{collection}-c2.md", collection, ["*"]) == 1
        assert {e.key for e in await store.find_entities(["mallory"], public, 5)} == {"mallory"}

        assert await store.delete_by_source(f"{collection}-c2.md", collection) == 1
        assert await store.find_entities(["mallory"], public, 5) == []  # orphan removed
        report = await store.quality_report(collection)
        assert report["counts"]["Chunk"] == 1 and report["counts"]["Entity"] == 2
    finally:
        await store.delete_collection(collection)
    report = await store.quality_report(collection)
    assert all(count == 0 for count in report["counts"].values())

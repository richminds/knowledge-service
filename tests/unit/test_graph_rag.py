"""Phase 4 (Part C): ontology, extraction, resolution, communities, graph search modes,
Cypher safety, fallback."""

import networkx as nx
import pytest

from app.core.config import RagMode
from app.rag.graph.base import ChunkGraph, Entity, GraphExport, GraphFact, Relation
from app.rag.graph.communities import (
    GraphMaintenance,
    detect_communities,
    duplicate_candidates,
    pagerank,
    personalized_pagerank,
)
from app.rag.graph.cypher import (
    CypherPlanner,
    CypherValidationError,
    load_templates,
    validate_cypher,
)
from app.rag.graph.extraction import GraphExtractor
from app.rag.graph.ontology import EntityResolver, load_ontology, load_resolver
from app.rag.ingestion.pipeline import LoadedDocument
from app.rag.strategies.graph import GraphOptions, GraphRagStrategy
from app.rag.types import RetrievalFilters

DEFAULT = RetrievalFilters(collection="default")


# -- ontology & resolution (C1, C3) --------------------------------------------------------
def test_ontology_remaps_synonyms_and_falls_back():
    ontology = load_ontology("config/ontology.yaml")
    assert ontology.entity_type("Department") == "Organization"
    assert ontology.entity_type("Spaceship") == "Concept"
    assert ontology.relation_type("leads") == "MANAGES"
    assert ontology.relation_type("orbits") == "RELATED_TO"


def test_resolver_canonicalises_aliases_and_legal_suffixes():
    resolver = load_resolver("config/entity_aliases.yaml")
    assert resolver.resolve("WBG").name == "World Bank Group"
    assert resolver.resolve("WBG").external_id == "TODO-mdm-0001"
    assert EntityResolver.normalise("The Acme Corp.") == "Acme"


async def test_extraction_validates_resolves_and_caches(chat_client):
    extractor = GraphExtractor(
        chat_client,
        ontology=load_ontology("config/ontology.yaml"),
        resolver=load_resolver("config/entity_aliases.yaml"),
        gleaning_rounds=1,
    )
    text = "Alice manages Finance. Finance funds Travel."
    entities, relations, _ = await extractor.extract_all(text)
    assert {e.key for e in entities} == {"alice", "finance", "travel"}
    assert all(r.type == "RELATED_TO" for r in relations)  # fake emits RELATED_TO
    calls = list(chat_client.json_calls)
    await extractor.extract_all(text)
    assert chat_client.json_calls == calls  # G-16: cached
    assert "graph_gleaning" in calls  # G-13


# -- communities & centrality (C4) -----------------------------------------------------------
def _export() -> GraphExport:
    export = GraphExport()
    groups = [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    for group in groups:
        for key in group:
            export.nodes[key] = Entity(key, "Concept")
        for x in group:
            for y in group:
                if x < y:
                    export.edges.append(GraphFact(x, "RELATED_TO", y, source_key=x, target_key=y))
    export.edges.append(
        GraphFact("a1", "RELATED_TO", "b1", source_key="a1", target_key="b1", weight=0.1)
    )
    return export


def test_louvain_finds_the_two_clusters_and_ppr_favours_seeds():
    from app.rag.graph.communities import to_networkx

    graph = to_networkx(_export())
    finest = detect_communities(graph)[0]
    assert sorted(sorted(c) for c in finest) == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    ranks = personalized_pagerank(graph, {"a2": 1.0})
    assert ranks["a3"] > ranks["b3"]
    assert abs(sum(pagerank(graph).values()) - 1.0) < 1e-6


def test_duplicate_candidates_same_type_only():
    export = GraphExport(
        nodes={
            "annual leave policy": Entity("Annual Leave Policy", "Policy"),
            "annual leave policies": Entity("Annual Leave Policies", "Policy"),
            "annual leave policy team": Entity("x", "Organization"),
        }
    )
    assert duplicate_candidates(export) == [("annual leave policy", ["annual leave policies"])]


async def test_community_reports_respect_acl_scope(
    graph_store, vector_store, embedding_client, chat_client
):
    def chunk(cid, acl, names):
        return ChunkGraph(
            cid,
            f"{cid}.md",
            "default",
            "text",
            acl=acl,
            entities=[Entity(n) for n in names],
            relations=[Relation(a, b) for a in names for b in names if a < b],
        )

    await graph_store.upsert_chunk_graph(chunk("pub", ["*"], ["Alpha", "Beta", "Gamma"]))
    await graph_store.upsert_chunk_graph(chunk("secret", ["grp-x"], ["Delta", "Epsilon", "Zeta"]))
    maintenance = GraphMaintenance(graph_store, chat_client, embedding_client, vector_store)
    result = await maintenance.build_communities("default")
    reports = [r for r in vector_store.records.values() if r.kind == "community"]
    assert result.communities == 1 and len(reports) == 1
    assert "Delta" not in reports[0].content and reports[0].acl == ["*"]
    assert graph_store.properties[("default", "alpha")]["community_0"] == 0


async def test_resolve_duplicates_merges_entities(graph_store, chat_client):
    await graph_store.upsert_chunk_graph(
        ChunkGraph(
            "c1",
            "s",
            "default",
            "t",
            entities=[
                Entity("Annual Leave Policy", "Policy"),
                Entity("Annual Leave Policies", "Policy"),
            ],
        )
    )
    merged = await GraphMaintenance(graph_store).resolve_duplicates("default")
    assert merged == 1


# -- graph search modes (C6) ---------------------------------------------------------------------
@pytest.fixture
async def org_graph(ingestion_pipeline):
    await ingestion_pipeline.run_for_documents(
        [
            LoadedDocument("org.md", "Alice manages Finance. Finance funds Travel."),
            LoadedDocument("ops.md", "Bob runs Procurement. Procurement supports Travel."),
        ],
        "default",
        targets=["vector", "graph"],
        metadata={"account_id": "acct-test"},
    )


def _strategy(
    graph_store, chat_client, embedding_client=None, vector_store=None, fallback=None, **opts
):
    from app.rag.graph.ontology import Ontology

    return GraphRagStrategy(
        graph_store,
        GraphExtractor(chat_client, ontology=Ontology()),
        options=GraphOptions(**opts),
        llm=chat_client,
        embedder=embedding_client,
        vector_store=vector_store,
        semantic_fallback=fallback,
        cypher=CypherPlanner(chat_client, load_templates("config/cypher_templates.yaml")),
    )


async def test_path_mode_connects_two_entities(org_graph, graph_store, chat_client):
    out = await _strategy(graph_store, chat_client).retrieve(
        "How is Alice related to Procurement?", DEFAULT, 5
    )
    modes = [s.detail["mode"] for s in out.trace if s.step == "graph_mode"]
    assert modes == ["path"]
    facts = out.contexts[0].content
    assert "Alice" in facts and "Procurement" in facts and "Travel" in facts
    assert out.contexts[0].metadata["supporting_chunks"]  # G-53


async def test_hybrid_mode_seeds_graph_from_vector_hits(
    org_graph, graph_store, vector_store, embedding_client, chat_client
):
    out = await _strategy(
        graph_store, chat_client, embedding_client, vector_store, mode="hybrid"
    ).retrieve("which department funds travel", DEFAULT, 4)
    retrievers = {c.metadata.get("retriever") for c in out.contexts}
    assert {"graph", "vector"} <= retrievers


async def test_global_mode_maps_community_reports(
    org_graph, graph_store, vector_store, embedding_client, chat_client
):
    await GraphMaintenance(
        graph_store, chat_client, embedding_client, vector_store, min_community_size=2
    ).build_communities("default")
    out = await _strategy(graph_store, chat_client, embedding_client, vector_store).retrieve(
        "What are the main themes across all departments?", DEFAULT, 3
    )
    assert [s.detail["mode"] for s in out.trace if s.step == "graph_mode"] == ["global"]
    assert out.contexts and all(c.metadata["retriever"] == "graph-global" for c in out.contexts)


async def test_cypher_template_results_are_acl_filtered(graph_store, chat_client):
    chat_client.cypher_choice = {"template": "who_manages", "params": {"entity": "finance"}}
    graph_store.cypher_rows = [
        ("Alice", "MANAGES", "Finance", "", ["*"], "c1", "acct-1"),
        ("Mallory", "MANAGES", "Finance", "", ["grp-secret"], "c2", "acct-1"),
        ("Eve", "MANAGES", "Finance", "", ["*"], "c3", "acct-2"),
    ]
    strategy = _strategy(graph_store, chat_client, mode="cypher")
    user = RetrievalFilters(
        collection="default", principals=frozenset({"*", "u1"}), account_id="acct-1"
    )
    out = await strategy.retrieve("Who manages Finance?", user, 5)
    content = out.contexts[0].content
    assert "Alice" in content and "Mallory" not in content and "Eve" not in content


async def test_freeform_cypher_disabled_under_acl(graph_store, chat_client):
    chat_client.cypher_generated = {
        "cypher": "MATCH (e:Entity {collection: $collection}) RETURN e.name",
        "columns": ["name"],
    }
    graph_store.cypher_rows = [("Secret",)]
    strategy = _strategy(graph_store, chat_client, mode="cypher")
    user = RetrievalFilters(collection="default", principals=frozenset({"*"}))
    out = await strategy.retrieve("List every entity", user, 5)
    assert all("Secret" not in c.content for c in out.contexts)
    out = await strategy.retrieve("List every entity", DEFAULT, 5)  # ACL not enforced
    assert "Secret" in out.contexts[0].content


async def test_falls_back_to_semantic_when_no_entities(
    ingestion_pipeline, graph_store, chat_client, strategies
):
    await ingestion_pipeline.run_for_documents(
        [LoadedDocument("leave.md", "annual leave is generous")],
        "default",
        metadata={"account_id": "acct-test"},
    )
    strategy = _strategy(
        graph_store, chat_client, fallback=strategies[RagMode.SEMANTIC], mode="local"
    )
    out = await strategy.retrieve("annual leave", DEFAULT, 3)
    assert out.contexts and any(s.step == "fallback_semantic" for s in out.trace)


async def test_similarity_linking_uses_entity_embeddings(
    org_graph, graph_store, embedding_client, chat_client
):
    entities = await _strategy(
        graph_store, chat_client, embedding_client, similarity_threshold=0.5
    )._link_entities("alice", [], DEFAULT, [])
    assert any(e.key == "alice" for e in entities)


# -- Cypher validation (G-44) ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n {collection: $collection}) DETACH DELETE n",
        "MATCH (n {collection: $collection}) SET n.x = 1 RETURN n",
        "CALL db.labels()",
        "MATCH (n) RETURN n",  # not scoped to the collection
        "MATCH (n {collection: $collection}) RETURN n; MATCH (m) DELETE m",
    ],
)
def test_validate_cypher_rejects_unsafe(query):
    with pytest.raises(CypherValidationError):
        validate_cypher(query)


def test_validate_cypher_caps_limit():
    assert validate_cypher("MATCH (n {collection: $collection}) RETURN n.name").endswith("LIMIT 50")
    assert validate_cypher("MATCH (n {collection: $collection}) RETURN n.name LIMIT 999").endswith(
        "LIMIT 50"
    )
    assert "SET" in validate_cypher(
        "MATCH (n {collection: $collection}) WHERE n.name = 'SET' RETURN n LIMIT 5"
    )


def test_pagerank_graph_helper_is_pure_python():
    graph = nx.path_graph(3)
    assert pagerank(graph)[1] > pagerank(graph)[0]


async def test_graph_knee_trims_low_ranked_facts(org_graph, graph_store, chat_client):
    question = "Who manages Finance?"
    full = await _strategy(graph_store, chat_client, mode="local").retrieve(question, DEFAULT, 5)
    trimmed = await _strategy(
        graph_store, chat_client, mode="local", knee=True, knee_sensitivity=0.0
    ).retrieve(question, DEFAULT, 5)

    knee_steps = [s for s in trimmed.trace if s.step == "graph_knee"]
    assert knee_steps and knee_steps[0].detail["kept"] <= knee_steps[0].detail["facts"]
    assert len(trimmed.contexts[0].content) <= len(full.contexts[0].content)
    assert not [s for s in full.trace if s.step == "graph_knee"]

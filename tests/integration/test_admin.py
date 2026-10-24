"""Admin RAG studio: role gate, config and scored evaluation runs."""

import pytest

from app.core.config import get_settings


@pytest.fixture
def admin_client(client, chat_client):
    from app.api import deps
    from app.main import app

    app.dependency_overrides[deps.get_chat_client] = lambda: chat_client
    client.post("/api/v1/ingestion", json={"source_uri": "docs", "collection": "default"})
    return client


def test_config_lists_modes_models_and_golden_set(admin_client):
    response = admin_client.get("/api/v1/admin/rag/config")

    assert response.status_code == 200, response.text
    body = response.json()
    settings = get_settings()
    assert set(body["modes"]["enabled"]) >= {"semantic", "graph", "agentic"}
    # The LLM gateway picks the model, so it is the only answer "model" on offer.
    assert body["answer_models"] == ["llm-gateway"]
    assert body["default_answer_model"] == "llm-gateway"
    assert body["models"]["embedding"] == f"llm-gateway ({settings.embedding_dimensions}d)"
    assert ".pdf" in body["ingestion"]["supported_extensions"]
    assert body["thresholds"]["all"]["hit"] == 0.7
    assert body["golden_set"][0]["id"] == "q1"


def test_evaluate_returns_answer_metrics_and_trace(admin_client):
    response = admin_client.post(
        "/api/v1/admin/rag/evaluate",
        json={
            "query": "How many leave days?",
            "rag_mode": "semantic",
            "top_k": 3,
            "expected_sources": ["leave.md"],
            "expected_answer": "25 days",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "llm-gateway"
    assert body["top_k"] == 3
    assert body["response"]["citations"][0]["source"].endswith("leave.md")
    assert body["response"]["trace"]
    retrieved = body["response"]["retrieved"]
    citations = body["response"]["citations"]
    assert [r["chunk_id"] for r in retrieved] == [c["chunk_id"] for c in citations]
    first = retrieved[0]
    assert first["content"].startswith(citations[0]["snippet"]) and first["kind"] == "chunk"
    assert first["doc_id"] and first["source"].endswith("leave.md")
    assert body["response"]["quality"]["judge"]["groundedness"] == 5
    metrics = body["metrics"]
    assert metrics["hit"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["faithfulness"] == 5.0
    assert "correctness" in metrics
    assert "routing_accuracy" not in metrics


def test_evaluate_rejects_unknown_model(admin_client):
    response = admin_client.post(
        "/api/v1/admin/rag/evaluate", json={"query": "leave", "model": "not-a-deployment"}
    )

    assert response.status_code == 422


def test_evaluate_overrides_mode_even_when_disabled(admin_client, retrieval_service, monkeypatch):
    monkeypatch.setattr(retrieval_service.registry, "_allow_override", False)

    assert admin_client.post("/api/v1/retrieval", json={"query": "leave", "rag_mode": "semantic"})
    response = admin_client.post(
        "/api/v1/admin/rag/evaluate", json={"query": "leave", "rag_mode": "semantic"}
    )

    assert response.status_code == 200, response.text


def test_evaluate_accepts_the_gateway_model(admin_client):
    response = admin_client.post(
        "/api/v1/admin/rag/evaluate", json={"query": "leave", "model": "llm-gateway"}
    )

    assert response.status_code == 200, response.text


def test_admin_endpoints_need_an_admin_gateway_user(admin_client, gateway, monkeypatch):
    url = "/api/v1/admin/rag/config"
    assert admin_client.get(url).status_code == 401
    assert admin_client.get(url, headers=gateway()).status_code == 403
    assert admin_client.get(url, headers=gateway(is_admin=True)).status_code == 200
    # Only the literal "true" counts.
    spoofed = {**gateway(), "X-Is-Admin": "yes"}
    assert admin_client.get(url, headers=spoofed).status_code == 403

    monkeypatch.setattr(get_settings(), "admin_requires_admin", False)
    assert admin_client.get(url, headers=gateway()).status_code == 200


def test_vector_store_status_counts_each_store(admin_client):
    body = admin_client.get("/api/v1/admin/rag/vector-stores?collection=default").json()

    assert body["collection"] == "default"
    [store] = body["stores"]
    assert store["default"] and store["count"] > 0 and not store["error"]
    assert body["in_sync"]


def test_vector_store_status_flags_a_store_that_fell_behind(admin_client, monkeypatch):
    from app.rag.vectorstore import factory
    from app.rag.vectorstore.multi import MultiVectorStore
    from tests.fakes import FakeVectorStore

    filled = factory._vector_store_singleton
    multi = MultiVectorStore(
        {"mongo": filled, "mongo_b": FakeVectorStore()}, default="mongo"
    )
    monkeypatch.setattr(factory, "_vector_store_singleton", multi)

    body = admin_client.get("/api/v1/admin/rag/vector-stores").json()

    counts = {s["name"]: s["count"] for s in body["stores"]}
    assert counts["mongo"] > 0 and counts["mongo_b"] == 0
    assert not body["in_sync"]


def test_vector_store_status_rejects_bad_collection(admin_client):
    assert admin_client.get("/api/v1/admin/rag/vector-stores?collection=../x").status_code == 422


def test_plain_chat_does_not_return_full_passages(admin_client):
    body = admin_client.post("/api/v1/chat", json={"query": "How many leave days?"}).json()

    assert body["citations"] and body.get("retrieved") is None

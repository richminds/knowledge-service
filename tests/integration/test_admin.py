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
    assert body["answer_models"][0] == settings.azure_openai_chat_deployment
    assert body["models"]["embedding"].startswith(settings.azure_openai_embedding_deployment)
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
    assert body["model"] == get_settings().azure_openai_chat_deployment
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


def test_admin_role_is_required_for_entra_users(admin_client, entra):
    assert admin_client.get("/api/v1/admin/rag/config", headers=entra()).status_code == 403
    assert (
        admin_client.get("/api/v1/admin/rag/config", headers=entra(roles=["sa_admin"])).status_code
        == 200
    )


def test_answer_model_override_is_scoped():
    from app.rag.generation.llm_client import ChatClient, answer_model

    client = ChatClient("https://example.openai.azure.com", "k", "2024-10-21", "gpt-4o")
    with answer_model("gpt-4o-mini"):
        assert client.answer_deployment == "gpt-4o-mini"
    assert client.answer_deployment == "gpt-4o"


def _recording_client(reject_max_tokens: bool):
    import httpx
    from openai import BadRequestError

    sent: list[dict] = []

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            if reject_max_tokens and "max_tokens" in kwargs:
                response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
                error = BadRequestError("unsupported", response=response, body=None)
                error.param = "max_tokens"
                raise error
            return "ok"

    client = type("C", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    return client, sent


async def test_reasoning_deployments_get_completion_token_parameters():
    from app.rag.generation.llm_client import REASONING_TOKEN_HEADROOM, ChatClient

    llm = ChatClient("https://example.openai.azure.com", "k", "2024-10-21", "gpt-5.5")
    llm._client, sent = _recording_client(reject_max_tokens=False)
    await llm._create("gpt-5.5", messages=[], max_tokens=300, temperature=0)
    await llm._create("gpt-4o", messages=[], max_tokens=300, temperature=0)

    assert sent[0] == {
        "model": "gpt-5.5",
        "messages": [],
        "max_completion_tokens": 300 + REASONING_TOKEN_HEADROOM,
    }
    assert sent[1]["max_tokens"] == 300 and sent[1]["temperature"] == 0


async def test_deployment_rejecting_max_tokens_is_learned():
    from app.rag.generation.llm_client import ChatClient

    llm = ChatClient("https://example.openai.azure.com", "k", "2024-10-21", "my-reasoner")
    llm._client, sent = _recording_client(reject_max_tokens=True)
    assert await llm._create("my-reasoner", messages=[], max_tokens=10, temperature=0) == "ok"
    await llm._create("my-reasoner", messages=[], max_tokens=10)

    assert [("max_tokens" in s) for s in sent] == [True, False, False]


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

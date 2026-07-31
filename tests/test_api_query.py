import pytest


@pytest.fixture()
def mock_rag_query(monkeypatch):
    async def _fake(question, metadata_filter=None, user_id=None, team_id=None):
        return {
            "answer": "This is the answer. [1]",
            "selected_chunks": [],
            "citation_validation": {
                "valid": True,
                "cited_numbers": [1],
                "missing_citations": False,
                "invalid_citations": [],
                "available_citations": [1],
            },
            "evaluation_metrics": {
                "faithfulness": 1.0,
                "context_precision": 1.0,
                "answer_relevance": 1.0,
                "evaluator": "local",
                "advanced_notes": "Local lexical metrics only.",
            },
        }

    monkeypatch.setattr("app.services.query_service.rag_query", _fake)
    return _fake


def test_query_endpoint_returns_pipeline_result(api, mock_rag_query):
    resp = api.post("/v1/query", json={"question": "What is the refund policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "This is the answer. [1]"
    assert body["citation_validation"]["valid"] is True
    assert body["evaluation_metrics"]["evaluator"] == "local"


def test_query_endpoint_rejects_overlong_question(api):
    resp = api.post("/v1/query", json={"question": "x" * 5000})
    assert resp.status_code == 413


def test_query_endpoint_requires_question_field(api):
    resp = api.post("/v1/query", json={})
    assert resp.status_code == 422

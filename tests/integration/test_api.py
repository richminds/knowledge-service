import json

import pytest


def test_health_endpoint_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_readiness_checks_vector_store(client):
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200, response.json()
    assert response.json()["checks"] == {"config": "ok", "vector_store": "ok"}


def test_request_id_is_propagated(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


def test_ingestion_job_then_retrieval_and_chat(client):
    start = client.post(
        "/api/v1/ingestion",
        json={
            "metadata": {"account_id": "acct-test"},
            "source_uri": "docs",
            "collection": "default",
        },
    )
    assert start.status_code == 202

    # TestClient runs background tasks before returning, so the job is already done.
    job = client.get(f"/api/v1/ingestion/{start.json()['job_id']}").json()
    assert job["status"] == "completed", job
    assert job["document_count"] == 2

    retrieval = client.post("/api/v1/retrieval", json={"query": "travel portal flights"})
    assert retrieval.status_code == 200
    assert retrieval.json()["results"][0]["source"].endswith("travel.txt")

    chat = client.post("/api/v1/chat", json={"query": "How many leave days?"})
    assert chat.status_code == 200
    assert chat.json()["citations"][0]["index"] == 1


def test_ingestion_rejects_path_traversal(client):
    start = client.post(
        "/api/v1/ingestion",
        json={
            "metadata": {"account_id": "acct-test"},
            "source_uri": "../../",
            "collection": "default",
        },
    )
    job = client.get(f"/api/v1/ingestion/{start.json()['job_id']}").json()

    assert job["status"] == "failed"
    assert "ingestion root" in job["detail"]


def test_invalid_collection_name_is_rejected(client):
    response = client.post("/api/v1/retrieval", json={"query": "q", "collection": "x' or '1"})

    assert response.status_code == 422


def test_upload_ingests_file(client):
    response = client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("notes.md", b"Quarterly budget review notes", "text/markdown")},
        data={
            "collection": "uploads",
            "metadata": json.dumps({"owner": "finance", "account_id": "acct-test"}),
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "source": "upload://notes.md",
        "collection": "uploads",
        "targets": ["vector"],
        "chunk_count": 1,
        "entity_count": 0,
        "relation_count": 0,
    }


def test_folder_upload_keeps_same_named_files_apart(client):
    for folder in ("hr", "it"):
        response = client.post(
            "/api/v1/ingestion/upload",
            files={"file": ("readme.txt", f"{folder} team readme".encode(), "text/plain")},
            data={
                "metadata": '{"account_id": "acct-test"}',
                "collection": "folders",
                "relative_path": f"Policies/{folder}/readme.txt",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["source"] == f"upload://Policies/{folder}/readme.txt"

    sources = {
        e["source"] for e in client.get("/api/v1/ingestion/manifest?collection=folders").json()
    }
    assert sources == {"upload://Policies/hr/readme.txt", "upload://Policies/it/readme.txt"}


def test_upload_path_cannot_escape_with_dot_dot(client):
    response = client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("a.txt", b"Some text worth indexing", "text/plain")},
        data={
            "metadata": '{"account_id": "acct-test"}',
            "collection": "folders",
            "relative_path": "../../etc/C:/a.txt",
        },
    )

    assert response.json()["source"] == "upload://etc/a.txt"


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("deck.pptx", b"PK\x03\x04 not really", "Unsupported file type .pptx"),
        ("tool.pdf", b"MZ\x90\x00 executable", "does not match the .pdf"),
        ("notes.md", b"%PDF-1.7 a pdf", "does not match the .md"),
        ("photo.png", b"\x89PNG\r\n\x1a\n", "Unsupported file type .png"),  # no image OCR
    ],
)
def test_upload_rejects_disallowed_or_mismatched_types(client, name, content, reason):
    response = client.post(
        "/api/v1/ingestion/upload",
        files={"file": (name, content)},
        data={"metadata": '{"account_id": "acct-test"}'},
    )

    assert response.status_code == 415
    assert reason in response.json()["detail"]


def test_delete_collection(client):
    client.post(
        "/api/v1/ingestion",
        json={"metadata": {"account_id": "acct-test"}, "source_uri": "docs", "collection": "temp"},
    )

    response = client.delete("/api/v1/ingestion/collections/temp")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] > 0


def test_chat_stream_emits_sse_events(client):
    client.post(
        "/api/v1/ingestion",
        json={
            "metadata": {"account_id": "acct-test"},
            "source_uri": "docs",
            "collection": "default",
        },
    )

    with client.stream("POST", "/api/v1/chat/stream", json={"query": "leave days"}) as response:
        body = "".join(response.iter_text())

    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event:")
    ]
    assert "step" in events  # live progress (A-42)
    assert [e for e in events if e != "step"][0] == "citations"
    assert "token" in events
    assert events[-1] == "done"


def test_gateway_identity_is_required_when_configured(client, gateway):
    assert client.post("/api/v1/retrieval", json={"query": "q"}).status_code == 401
    ok = client.post("/api/v1/retrieval", json={"query": "q"}, headers=gateway())
    assert ok.status_code == 200
    # Health probes stay open for the platform.
    assert client.get("/api/v1/health").status_code == 200


def test_rag_modes_endpoint(client):
    body = client.get("/api/v1/rag/modes").json()

    assert body == {
        "default": "semantic",
        "enabled": ["semantic", "hybrid", "hybrid_graph", "graph", "agentic"],
        "allow_override": True,
        "vector_stores": [],
        "default_vector_store": "",
    }


def test_each_rag_mode_can_be_selected_per_request(client):
    upload = client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("org.md", b"Alice manages the Finance department.", "text/markdown")},
        data={"metadata": '{"account_id": "acct-test"}', "targets": "vector,graph"},
    )
    assert upload.json()["entity_count"] >= 2, upload.text

    for mode in ("semantic", "hybrid", "hybrid_graph", "graph", "agentic"):
        response = client.post(
            "/api/v1/chat",
            json={"query": "Who manages Finance?", "rag_mode": mode, "include_trace": True},
        )
        body = response.json()
        assert response.status_code == 200, body
        assert body["rag_mode"] == mode
        assert body["citations"], mode
        assert body["trace"], mode


def test_trace_is_omitted_unless_requested(client):
    body = client.post("/api/v1/retrieval", json={"query": "leave"}).json()

    assert body["rag_mode"] == "semantic"
    assert body["trace"] is None


def test_unknown_rag_mode_is_a_validation_error(client):
    response = client.post("/api/v1/chat", json={"query": "q", "rag_mode": "magic"})

    assert response.status_code == 422


def _ingest_docs(client):
    client.post(
        "/api/v1/ingestion",
        json={
            "metadata": {"account_id": "acct-test"},
            "source_uri": "docs",
            "collection": "default",
        },
    )


def test_chat_returns_quality_usage_and_response_id(client):
    _ingest_docs(client)

    response = client.post("/api/v1/chat", json={"query": "How many leave days?"})
    body = response.json()

    assert body["response_id"] == response.headers["x-request-id"]
    assert body["quality"]["confidence"] == "high"
    assert body["quality"]["evaluated_by"] == "signals"
    assert body["quality"]["explanation"]
    assert body["citations"][0]["cited"] is True
    assert body["usage"] is not None
    assert body["latency_ms"] > 0


def test_chat_with_evaluate_includes_judge_scores(client):
    _ingest_docs(client)

    body = client.post(
        "/api/v1/chat", json={"query": "How many leave days?", "evaluate": True}
    ).json()

    assert body["quality"]["evaluated_by"] == "signals+llm_judge"
    assert body["quality"]["judge"]["groundedness"] == 5


def test_chat_without_context_is_low_confidence(client):
    body = client.post("/api/v1/chat", json={"query": "anything", "collection": "empty"}).json()

    assert body["quality"]["confidence"] == "low"
    assert body["quality"]["signals"]["answered"] is False


def test_chat_stream_sends_quality_before_done(client):
    _ingest_docs(client)

    with client.stream(
        "POST", "/api/v1/chat/stream", json={"query": "leave days", "evaluate": True}
    ) as response:
        body = "".join(response.iter_text())

    events = [
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event:")
    ]
    assert events[-2:] == ["quality", "done"]
    assert '"evaluated_by": "signals+llm_judge"' in body


def test_feedback_is_accepted(client):
    response = client.post(
        "/api/v1/feedback",
        json={
            "response_id": "abc123",
            "rating": "down",
            "comment": "Wrong year",
            "rag_mode": "graph",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "recorded"}


def test_manifest_lists_document_status(client):
    client.post(
        "/api/v1/ingestion",
        json={
            "metadata": {"account_id": "acct-test"},
            "source_uri": "docs",
            "collection": "default",
        },
    )

    entries = client.get("/api/v1/ingestion/manifest", params={"collection": "default"}).json()

    statuses = {e["source"].replace("\\", "/").rsplit("/", 1)[-1]: e["status"] for e in entries}
    assert statuses["leave.md"] == "indexed" and statuses["travel.txt"] == "indexed"
    assert statuses["ignored.bin"] == "quarantined"
    quarantined = client.get(
        "/api/v1/ingestion/manifest", params={"collection": "default", "status": "quarantined"}
    ).json()
    assert [e["error"] for e in quarantined] == ["no_or_little_text"]

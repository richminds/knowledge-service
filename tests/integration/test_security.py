"""Authentication, ACL enforcement over HTTP, rate limiting, deletion endpoints."""

from app.core.config import get_settings


def _upload(client, name, text, acl="", headers=None):
    return client.post(
        "/api/v1/ingestion/upload",
        files={"file": (name, text.encode(), "text/markdown")},
        data={"acl": acl},
        headers=headers or {},
    )


def test_gateway_identity_drives_acl(client, gateway):
    # Documents shared with a user id or an account id; the gateway asserts both per request.
    assert (
        _upload(client, "mine.md", "Payroll schedule for alice", "alice", gateway()).status_code
        == 200
    )
    assert (
        _upload(client, "hr.md", "Payroll schedule for HR staff", "ACC-hr", gateway()).status_code
        == 200
    )
    assert (
        _upload(
            client, "pub.md", "Payroll questions go to the service desk", "", gateway()
        ).status_code
        == 200
    )

    def sources(headers):
        response = client.post("/api/v1/retrieval", json={"query": "payroll"}, headers=headers)
        return {r["source"] for r in response.json()["results"]}

    assert sources(gateway(user_id="bob")) == {"upload://pub.md"}
    assert sources(gateway(user_id="alice")) == {"upload://mine.md", "upload://pub.md"}
    assert sources(gateway(user_id="bob", account_id="ACC-hr")) == {
        "upload://hr.md",
        "upload://pub.md",
    }
    # Admin is a gate for endpoints, not an ACL bypass.
    assert sources(gateway(user_id="bob", is_admin=True)) == {"upload://pub.md"}


def test_missing_gateway_identity_is_rejected(client, gateway):
    assert client.post("/api/v1/retrieval", json={"query": "q"}).status_code == 401
    # Other identity headers without X-User-ID do not make a caller.
    partial = {"X-User-Email": "u@corp.org", "X-Account-ID": "ACC-1", "X-Is-Admin": "true"}
    assert client.post("/api/v1/retrieval", json={"query": "q"}, headers=partial).status_code == 401
    blank = {"X-User-ID": "   "}
    assert client.post("/api/v1/retrieval", json={"query": "q"}, headers=blank).status_code == 401
    # A bearer token alone is not validated here, so it is not an identity either.
    bearer = {"Authorization": "Bearer anything"}
    assert client.post("/api/v1/retrieval", json={"query": "q"}, headers=bearer).status_code == 401
    assert client.get("/api/v1/health").status_code == 200  # probes stay open


def test_ingestion_needs_admin_only_when_configured(client, gateway, monkeypatch):
    # Default: any signed-in user may ingest.
    assert _upload(client, "a.md", "text", headers=gateway()).status_code == 200

    monkeypatch.setattr(get_settings(), "ingestion_requires_admin", True)
    assert _upload(client, "a.md", "text", headers=gateway()).status_code == 403
    assert _upload(client, "a.md", "text", headers=gateway(is_admin=True)).status_code == 200
    gone = client.delete(
        "/api/v1/ingestion/sources", params={"source": "upload://a.md"}, headers=gateway()
    )
    assert gone.status_code == 403


def test_auth_mode_none_does_not_enforce_acls(client):
    # Local development only: the anonymous caller sees everything.
    _upload(client, "restricted.md", "Board minutes", "grp-board")
    results = client.post("/api/v1/retrieval", json={"query": "Board minutes"})
    assert [r["source"] for r in results.json()["results"]] == ["upload://restricted.md"]


def test_rate_limit_returns_429(client, monkeypatch):
    from app.core import ratelimit

    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    monkeypatch.setattr(ratelimit, "_limiter", None)
    codes = [client.post("/api/v1/retrieval", json={"query": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    monkeypatch.setattr(ratelimit, "_limiter", None)


def test_delete_source_removes_document_everywhere(client, vector_store):
    _upload(client, "gone.md", "Temporary notice about parking")
    assert any(r.source == "upload://gone.md" for r in vector_store.records.values())

    response = client.delete("/api/v1/ingestion/sources", params={"source": "upload://gone.md"})

    assert response.status_code == 200 and response.json()["deleted_chunks"] >= 1
    assert not any(r.source == "upload://gone.md" for r in vector_store.records.values())


def test_delete_my_conversations(client):
    _upload(client, "a.md", "Annual leave is 25 days")
    client.post("/api/v1/chat", json={"query": "annual leave"})
    response = client.delete("/api/v1/conversations/me")
    assert response.json() == {"deleted_conversations": 1}

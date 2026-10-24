"""Authentication, ACL enforcement over HTTP, rate limiting, deletion endpoints."""

from pydantic import SecretStr

from app.core.config import get_settings


def _upload(client, name, text, acl="", headers=None):
    return client.post(
        "/api/v1/ingestion/upload",
        files={"file": (name, text.encode(), "text/markdown")},
        data={"acl": acl},
        headers=headers or {},
    )


def test_entra_token_principals_drive_acl(client, entra):
    assert (
        _upload(client, "hr.md", "Payroll schedule for HR staff", "grp-hr", entra()).status_code
        == 200
    )
    assert (
        _upload(
            client, "pub.md", "Payroll questions go to the service desk", "", entra()
        ).status_code
        == 200
    )

    outsider = client.post("/api/v1/retrieval", json={"query": "payroll"}, headers=entra())
    insider = client.post(
        "/api/v1/retrieval", json={"query": "payroll"}, headers=entra(groups=["grp-hr"])
    )

    assert {r["source"] for r in outsider.json()["results"]} == {"upload://pub.md"}
    assert {r["source"] for r in insider.json()["results"]} == {"upload://hr.md", "upload://pub.md"}


def test_invalid_or_missing_tokens_are_rejected(client, entra):
    assert client.post("/api/v1/retrieval", json={"query": "q"}).status_code == 401
    bad_audience = entra(audience="api://someone-else")
    assert (
        client.post("/api/v1/retrieval", json={"query": "q"}, headers=bad_audience).status_code
        == 401
    )
    expired = entra(expires_in=-120)
    assert client.post("/api/v1/retrieval", json={"query": "q"}, headers=expired).status_code == 401
    assert client.get("/api/v1/health").status_code == 200  # probes stay open


def test_ingestion_role_is_required_when_configured(client, entra, monkeypatch):
    monkeypatch.setattr(get_settings(), "ingestion_required_role", "Ingestion.Write")
    assert _upload(client, "a.md", "text", headers=entra()).status_code == 403
    assert (
        _upload(client, "a.md", "text", headers=entra(roles=["Ingestion.Write"])).status_code == 200
    )


def test_api_key_callers_get_configured_principals(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_mode", "api_key")
    monkeypatch.setattr(settings, "api_key", SecretStr("k"))
    headers = {"X-API-Key": "k"}
    _upload(client, "restricted.md", "Board minutes", "grp-board", headers)

    results = client.post("/api/v1/retrieval", json={"query": "Board minutes"}, headers=headers)
    assert results.json()["results"] == []  # default API_KEY_PRINCIPALS = ["*"]

    monkeypatch.setattr(settings, "api_key_principals", ["*", "grp-board"])
    results = client.post("/api/v1/retrieval", json={"query": "Board minutes"}, headers=headers)
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

"""Authentication, ACL enforcement over HTTP, rate limiting, deletion endpoints."""

from app.core.config import get_settings


def _upload(client, name, text, acl="", headers=None):
    return client.post(
        "/api/v1/ingestion/upload",
        files={"file": (name, text.encode(), "text/markdown")},
        data={"acl": acl},
        headers=headers or {},
    )


def _chunk_metadata(vector_store, source):
    return [r.metadata for r in vector_store.records.values() if r.source == source]


def test_ingested_chunks_carry_the_callers_account_id(client, gateway, vector_store):
    # A caller-supplied account_id is replaced by the one the gateway asserts.
    response = client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("acc.md", b"Travel policy for the sales team", "text/markdown")},
        data={"metadata": '{"account_id": "ACC-spoofed", "team": "sales"}'},
        headers=gateway(account_id="ACC-sales"),
    )
    assert response.status_code == 200

    metadata = _chunk_metadata(vector_store, "upload://acc.md")
    assert metadata
    assert all(m["account_id"] == "ACC-sales" and m["team"] == "sales" for m in metadata)


def test_callers_without_an_account_cannot_set_one(client, gateway, vector_store):
    response = client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("none.md", b"Office opening hours", "text/markdown")},
        data={"metadata": '{"account_id": "ACC-spoofed"}'},
        headers=gateway(),
    )
    assert response.status_code == 200

    metadata = _chunk_metadata(vector_store, "upload://none.md")
    assert metadata
    assert all("account_id" not in m for m in metadata)


def _sources(client, headers, query="payroll"):
    response = client.post("/api/v1/retrieval", json={"query": query}, headers=headers)
    return {r["source"] for r in response.json()["results"]}


def test_gateway_identity_drives_acl(client, gateway):
    # Within one account, a document's ACL still decides which users read it.
    uploader = gateway(user_id="alice", account_id="ACC-1")
    assert (
        _upload(client, "mine.md", "Payroll schedule for alice", "alice", uploader).status_code
        == 200
    )
    assert (
        _upload(
            client, "pub.md", "Payroll questions go to the service desk", "", uploader
        ).status_code
        == 200
    )

    assert _sources(client, gateway(user_id="bob", account_id="ACC-1")) == {"upload://pub.md"}
    assert _sources(client, gateway(user_id="alice", account_id="ACC-1")) == {
        "upload://mine.md",
        "upload://pub.md",
    }
    # Admin is a gate for endpoints, not an ACL bypass.
    assert _sources(client, gateway(user_id="bob", account_id="ACC-1", is_admin=True)) == {
        "upload://pub.md"
    }


def test_retrieval_is_scoped_to_the_callers_account(client, gateway):
    # Shared with everyone ("*"), but only inside the account it was ingested from.
    _upload(client, "a.md", "Payroll runs on the 25th", "", gateway(account_id="ACC-a"))
    _upload(client, "b.md", "Payroll runs on the 1st", "", gateway(account_id="ACC-b"))
    # An ACL naming another account does not let that account in.
    _upload(client, "c.md", "Payroll cut-off dates", "ACC-b", gateway(account_id="ACC-a"))

    assert _sources(client, gateway(account_id="ACC-a")) == {"upload://a.md"}
    assert _sources(client, gateway(account_id="ACC-b")) == {"upload://b.md"}
    # No account: only content ingested without one.
    assert _sources(client, gateway()) == set()


def test_account_isolation_can_be_turned_off(client, gateway, monkeypatch):
    monkeypatch.setattr(get_settings(), "account_isolation", False)
    _upload(client, "a.md", "Payroll runs on the 25th", "", gateway(account_id="ACC-a"))

    assert _sources(client, gateway(account_id="ACC-b")) == {"upload://a.md"}


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

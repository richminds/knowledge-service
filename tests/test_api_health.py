def test_root_banner(api):
    resp = api.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "knowledge-service"


def test_liveness_never_touches_a_dependency(api):
    resp = api.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_reports_not_ready_without_mongo(api):
    # RAG_MONGO_URI is "" in the test environment (see conftest.py).
    resp = api.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    names = {d["name"] for d in body["dependencies"]}
    assert {"mongodb", "llm_gateway"} <= names


def test_health_alias_matches_ready(api):
    resp = api.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"

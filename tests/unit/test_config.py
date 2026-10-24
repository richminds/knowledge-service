import pytest
from pydantic import ValidationError

from app.core.config import RagMode, Settings


def test_list_settings_accept_comma_separated_and_json(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED_MODES", "semantic, graph")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://a.org","https://b.org"]')

    settings = Settings()

    assert settings.rag_enabled_modes == [RagMode.SEMANTIC, RagMode.GRAPH]
    assert settings.cors_allow_origins == ["https://a.org", "https://b.org"]


def test_urls_are_validated_and_normalised(monkeypatch):
    monkeypatch.setenv("API_GATEWAY_URL", "http://gateway.test:8000/")
    assert Settings().api_gateway_url == "http://gateway.test:8000"

    monkeypatch.setenv("LLM_GATEWAY_URL", "llm-gateway.test")
    with pytest.raises(ValidationError, match="http"):
        Settings()


def test_secrets_are_masked_in_repr(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "super-secret")

    settings = Settings()

    assert "super-secret" not in repr(settings)
    assert settings.llm_gateway_api_key.get_secret_value() == "super-secret"


def test_a_gateway_url_is_required_for_llm_calls(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", "")
    monkeypatch.setenv("API_GATEWAY_URL", "")
    message = "API_GATEWAY_URL (or LLM_GATEWAY_URL) is required for LLM calls"

    assert message in Settings().configuration_errors()

    monkeypatch.setenv("API_GATEWAY_URL", "http://gateway.test")
    assert message not in Settings().configuration_errors()


def test_auth_defaults_require_admin_for_the_studio_only():
    settings = Settings()

    assert settings.auth_mode == "none"
    assert settings.admin_requires_admin and not settings.ingestion_requires_admin
    with pytest.raises(ValidationError):
        Settings(auth_mode="entra")


def test_configuration_errors_reflect_enabled_modes(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED_MODES", "semantic,graph,agentic")
    monkeypatch.setenv("RAG_MODE", "agentic")
    monkeypatch.setenv("MONGODB_URL", "")

    errors = Settings().configuration_errors()

    assert errors == [
        "MONGODB_URL is required for semantic/hybrid/agentic(vector) RAG",
        "MONGODB_URL is required for graph RAG",
    ]


def test_default_mode_must_be_enabled(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "graph")

    assert "RAG_MODE=graph is not in RAG_ENABLED_MODES" in Settings().configuration_errors()


def test_vector_providers_default_first_and_deduplicated(monkeypatch):
    assert Settings().active_vector_providers == ["mongo"]

    monkeypatch.setenv("VECTOR_STORE_PROVIDERS", "mongo, mongo")

    assert Settings().active_vector_providers == ["mongo"]


def test_removed_vector_providers_are_rejected(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "pgvector")

    with pytest.raises(ValidationError):
        Settings()


def test_mongodb_url_is_required_for_the_stores(monkeypatch):
    monkeypatch.setenv("MONGODB_URL", "")
    monkeypatch.setenv("MANIFEST_STORE", "mongo")

    settings = Settings()

    assert not settings.vector_store_configured
    assert "MONGODB_URL is required for MANIFEST_STORE=mongo" in settings.configuration_errors()

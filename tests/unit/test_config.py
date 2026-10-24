from types import SimpleNamespace

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
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    assert Settings().azure_openai_endpoint == "https://x.openai.azure.com"

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "x.openai.azure.com")
    with pytest.raises(ValidationError, match="http"):
        Settings()


def test_secrets_are_masked_in_repr(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "super-secret")

    settings = Settings()

    assert "super-secret" not in repr(settings)
    assert settings.azure_openai_api_key.get_secret_value() == "super-secret"


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


class _FakeSecretClient:
    """Mimics azure.keyvault.secrets.SecretClient for the settings source."""

    secrets = {"azure-openai-api-key": "kv-openai-key", "mongodb-url": "mongodb://kv-host"}

    def __init__(self, vault_url, credential):
        self.vault_url = vault_url

    def list_properties_of_secrets(self):
        return [SimpleNamespace(name=n, enabled=True) for n in self.secrets]

    def get_secret(self, name):
        return SimpleNamespace(value=self.secrets[name])


def test_key_vault_secrets_fill_settings_but_env_wins(monkeypatch):
    import azure.keyvault.secrets

    from app.core import azure_auth

    # pydantic-settings imports SecretClient from this module when the source is built.
    monkeypatch.setattr(azure.keyvault.secrets, "SecretClient", _FakeSecretClient)
    monkeypatch.setattr(azure_auth, "get_sync_azure_credential", lambda: object())
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://kv-test.vault.azure.net")
    monkeypatch.setenv("MONGODB_URL", "mongodb://from-env")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")  # blank must not mask the vault value

    settings = Settings()

    assert settings.azure_openai_api_key.get_secret_value() == "kv-openai-key"
    assert settings.mongodb_url.get_secret_value() == "mongodb://from-env"


def test_mongodb_url_is_read_from_key_vault(monkeypatch):
    import azure.keyvault.secrets

    from app.core import azure_auth

    monkeypatch.setattr(azure.keyvault.secrets, "SecretClient", _FakeSecretClient)
    monkeypatch.setattr(azure_auth, "get_sync_azure_credential", lambda: object())
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://kv-test.vault.azure.net")

    monkeypatch.delenv("MONGODB_URL", raising=False)

    assert Settings().mongodb_url.get_secret_value() == "mongodb://kv-host"

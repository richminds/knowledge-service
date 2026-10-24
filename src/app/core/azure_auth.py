from functools import lru_cache

from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


@lru_cache
def get_azure_credential() -> DefaultAzureCredential:
    """Process-wide async credential: Managed Identity in Azure, az login / VS Code locally."""
    return DefaultAzureCredential()


@lru_cache
def get_sync_azure_credential() -> SyncDefaultAzureCredential:
    """Sync credential, used only while loading settings from Key Vault at startup."""
    return SyncDefaultAzureCredential()


def build_openai_client(endpoint: str, api_key: str, api_version: str) -> AsyncAzureOpenAI:
    """Use the API key when one is configured, otherwise Entra ID (keyless) auth."""
    if api_key:
        return AsyncAzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    token_provider = get_bearer_token_provider(get_azure_credential(), COGNITIVE_SERVICES_SCOPE)
    return AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

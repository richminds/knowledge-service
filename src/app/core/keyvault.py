"""Load secrets from Azure Key Vault into Settings.

Naming convention: the Key Vault secret name is the setting name in lower-case with
dashes, because Key Vault does not allow underscores:

    AZURE_OPENAI_API_KEY  <-  secret "azure-openai-api-key"
    MONGODB_URL           <-  secret "mongodb-url"
    API_KEY               <-  secret "api-key"

Only secrets whose names match a setting are read. The app identity needs the
"Key Vault Secrets User" role (granted by infra/main.bicep).
"""

from collections.abc import Mapping

from azure.core.credentials import TokenCredential
from pydantic_settings import AzureKeyVaultSettingsSource, BaseSettings


class KeyVaultSettingsSource(AzureKeyVaultSettingsSource):
    def __init__(
        self, settings_cls: type[BaseSettings], url: str, credential: TokenCredential
    ) -> None:
        super().__init__(settings_cls, url, credential, dash_to_underscore=True)

    def _load_env_vars(self) -> Mapping[str, str | None]:
        try:
            return super()._load_env_vars()
        except Exception as exc:  # noqa: BLE001 - fail fast with an actionable message
            raise RuntimeError(
                f"Could not read secrets from Key Vault {self._url}: {exc}. Check "
                "AZURE_KEY_VAULT_URL, network access, and that this identity has the "
                "'Key Vault Secrets User' role."
            ) from exc

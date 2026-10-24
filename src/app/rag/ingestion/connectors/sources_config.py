"""Named source definitions (config/sources.yaml) with secret references (C-064).

Values may reference secrets as:
  ${ENV_VAR}          - environment variable / App Service setting
  ${kv:secret-name}   - Azure Key Vault secret (AZURE_KEY_VAULT_URL, managed identity)
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"\$\{(kv:)?([A-Za-z0-9_\-]+)\}")


@lru_cache(maxsize=64)
def _key_vault_secret(name: str) -> str:
    from azure.keyvault.secrets import SecretClient

    from app.core.azure_auth import get_sync_azure_credential
    from app.core.config import get_settings

    vault = get_settings().azure_key_vault_url
    if not vault:
        raise ValueError(f"${{kv:{name}}} used but AZURE_KEY_VAULT_URL is not set")
    return SecretClient(vault, get_sync_azure_credential()).get_secret(name).value or ""


def resolve_secrets(value: Any) -> Any:
    if isinstance(value, str):
        return _REF.sub(
            lambda m: (
                _key_vault_secret(m.group(2)) if m.group(1) else os.environ.get(m.group(2), "")
            ),
            value,
        )
    if isinstance(value, dict):
        return {k: resolve_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_secrets(v) for v in value]
    return value


@lru_cache(maxsize=4)
def load_sources(path: str) -> dict[str, dict]:
    file = Path(path)
    if not file.is_file():
        return {}
    return yaml.safe_load(file.read_text(encoding="utf-8")) or {}


def source_definition(path: str, kind: str, name: str) -> dict:
    definition = (load_sources(path).get(kind) or {}).get(name)
    if definition is None:
        raise ValueError(f"No {kind} source named {name!r} in {path}")
    return resolve_secrets(definition)

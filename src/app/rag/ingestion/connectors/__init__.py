"""Connector registry: picks a connector from the source URI scheme (C-057).

  <path> | file://        LocalConnector       (INGESTION_LOCAL_ROOT)
  blob://container/prefix  BlobConnector        (AZURE_STORAGE_*)
  sharepoint://host/sites/... SharePointConnector (SHAREPOINT_*)
  confluence://SPACE        ConfluenceConnector  (CONFLUENCE_*)
  db://<name>               DatabaseConnector    (config/sources.yaml)
  api://<name>              RestApiConnector     (config/sources.yaml)
  https://...               WebConnector         (WEB_CRAWL_*)
Connectors whose settings are blank are not built; using their scheme raises a clear error.
"""

from collections.abc import Callable

from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.connectors.base import Connector

ConnectorFactory = Callable[[], Connector]


class ConnectorRegistry:
    def __init__(
        self, local: Connector, factories: dict[str, ConnectorFactory] | None = None
    ) -> None:
        self._local = local
        self._factories = factories or {}
        self._built: dict[str, Connector] = {}

    def register(self, scheme: str, factory: ConnectorFactory) -> None:
        self._factories[scheme] = factory

    def for_uri(self, uri: str) -> Connector:
        for scheme, factory in self._factories.items():
            if uri.startswith(scheme):
                if scheme not in self._built:
                    try:
                        self._built[scheme] = factory()
                    except DocumentLoadError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise DocumentLoadError(
                            f"Connector for {scheme} is not configured: {exc}"
                        ) from exc
                return self._built[scheme]
        if "://" in uri and not uri.startswith("file://"):
            raise DocumentLoadError(f"No connector configured for {uri.split('://')[0]}://")
        return self._local

    async def close(self) -> None:
        for connector in self._built.values():
            await connector.close()

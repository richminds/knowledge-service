"""Shared MongoDB client (MongoDB Atlas): one connection pool per URL for the process.

Used by the vector store, the graph store and the ingestion manifest. Vector and keyword
search need Atlas Search indexes ($vectorSearch / $search), so MONGODB_URL must point at an
Atlas cluster (or a local Atlas deployment: `atlas deployments setup --type local`).
"""

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

_clients: dict[str, AsyncMongoClient] = {}


def get_mongo_client(url: str) -> AsyncMongoClient:
    if not url:
        raise ValueError("MONGODB_URL is not configured")
    client = _clients.get(url)
    if client is None:
        client = _clients[url] = AsyncMongoClient(url, appname="knowledge-service", tz_aware=True)
    return client


def get_mongo_database(url: str, name: str) -> AsyncDatabase:
    return get_mongo_client(url)[name]


async def close_mongo_clients() -> None:
    """Close every pooled client (app shutdown). Stores share clients, so they never close one."""
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        await client.close()

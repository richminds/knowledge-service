"""Conversation history (C-157) and persisted turns (C-191).

CONVERSATION_STORE:
  memory - per-process, bounded (default; history is lost on restart)
  cosmos - Azure Cosmos DB for NoSQL. Placeholders: COSMOS_ENDPOINT, COSMOS_DATABASE,
           COSMOS_CONVERSATION_CONTAINER (partition key /user_id). Auth: managed identity
           (Cosmos DB Built-in Data Contributor) or COSMOS_KEY.
Conversations are scoped to the user: a conversation id is only readable by its owner.
"""

from collections import OrderedDict
from datetime import UTC, datetime
from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationStore(Protocol):
    async def history(
        self, conversation_id: str, user_id: str, max_turns: int
    ) -> list[tuple[str, str]]: ...

    async def append(
        self,
        conversation_id: str,
        user_id: str,
        question: str,
        answer: str,
        *,
        response_id: str,
        rag_mode: str,
        sources: list[str],
    ) -> None: ...

    async def delete_user(self, user_id: str) -> int: ...


def _turn(question: str, answer: str, response_id: str, rag_mode: str, sources: list[str]) -> dict:
    return {
        "at": datetime.now(UTC).isoformat(),
        "question": question,
        "answer": answer,
        "response_id": response_id,
        "rag_mode": rag_mode,
        "sources": sources,
    }


def _as_history(turns: list[dict], max_turns: int) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    for turn in turns[-max_turns:]:
        history += [("user", turn["question"]), ("assistant", turn["answer"])]
    return history


class InMemoryConversationStore:
    def __init__(self, max_conversations: int = 10_000, max_turns_kept: int = 50) -> None:
        self._data: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
        self._max_conversations = max_conversations
        self._max_turns_kept = max_turns_kept

    async def history(
        self, conversation_id: str, user_id: str, max_turns: int
    ) -> list[tuple[str, str]]:
        return _as_history(self._data.get((user_id, conversation_id), []), max_turns)

    async def append(
        self, conversation_id, user_id, question, answer, *, response_id, rag_mode, sources
    ) -> None:
        key = (user_id, conversation_id)
        turns = self._data.pop(key, [])
        turns.append(_turn(question, answer, response_id, rag_mode, sources))
        self._data[key] = turns[-self._max_turns_kept :]
        while len(self._data) > self._max_conversations:
            self._data.popitem(last=False)

    async def delete_user(self, user_id: str) -> int:
        keys = [k for k in self._data if k[0] == user_id]
        for key in keys:
            del self._data[key]
        return len(keys)


class CosmosConversationStore:
    def __init__(
        self,
        endpoint: str,
        database: str,
        container: str,
        credential,
        max_turns_kept: int = 50,
    ) -> None:
        from azure.cosmos.aio import CosmosClient

        self._client = CosmosClient(endpoint, credential=credential)
        self._container = self._client.get_database_client(database).get_container_client(container)
        self._max_turns_kept = max_turns_kept

    async def _read(self, conversation_id: str, user_id: str) -> dict | None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return await self._container.read_item(conversation_id, partition_key=user_id)
        except CosmosResourceNotFoundError:
            return None

    async def history(
        self, conversation_id: str, user_id: str, max_turns: int
    ) -> list[tuple[str, str]]:
        item = await self._read(conversation_id, user_id)
        return _as_history(item["turns"], max_turns) if item else []

    async def append(
        self, conversation_id, user_id, question, answer, *, response_id, rag_mode, sources
    ) -> None:
        item = await self._read(conversation_id, user_id) or {
            "id": conversation_id,
            "user_id": user_id,
            "turns": [],
        }
        item["turns"] = [*item["turns"], _turn(question, answer, response_id, rag_mode, sources)][
            -self._max_turns_kept :
        ]
        await self._container.upsert_item(item)

    async def delete_user(self, user_id: str) -> int:
        deleted = 0
        query = "SELECT c.id FROM c WHERE c.user_id = @user"
        async for item in self._container.query_items(
            query, parameters=[{"name": "@user", "value": user_id}], partition_key=user_id
        ):
            await self._container.delete_item(item["id"], partition_key=user_id)
            deleted += 1
        return deleted

    async def close(self) -> None:
        await self._client.close()

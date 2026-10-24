"""Conversation history (C-157) and persisted turns (C-191).

CONVERSATION_STORE:
  memory - per-process, bounded (default; history is lost on restart)
  mongo  - MongoDB (MONGODB_URL / MONGODB_DATABASE), one document per conversation in
           MONGODB_CONVERSATIONS_COLLECTION, keyed by (user_id, conversation_id)
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


class MongoConversationStore:
    def __init__(self, collection, max_turns_kept: int = 50) -> None:
        self._collection = collection  # pymongo AsyncCollection
        self._max_turns_kept = max_turns_kept
        self._indexed = False

    async def _ensure_index(self) -> None:
        if not self._indexed:
            await self._collection.create_index(
                [("user_id", 1), ("conversation_id", 1)], unique=True
            )
            self._indexed = True

    async def history(
        self, conversation_id: str, user_id: str, max_turns: int
    ) -> list[tuple[str, str]]:
        item = await self._collection.find_one(
            {"user_id": user_id, "conversation_id": conversation_id}, {"turns": 1}
        )
        return _as_history(item["turns"], max_turns) if item else []

    async def append(
        self, conversation_id, user_id, question, answer, *, response_id, rag_mode, sources
    ) -> None:
        await self._ensure_index()
        await self._collection.update_one(
            {"user_id": user_id, "conversation_id": conversation_id},
            {
                "$push": {
                    "turns": {
                        "$each": [_turn(question, answer, response_id, rag_mode, sources)],
                        "$slice": -self._max_turns_kept,
                    }
                },
                "$set": {"updated_at": datetime.now(UTC)},
            },
            upsert=True,
        )

    async def delete_user(self, user_id: str) -> int:
        result = await self._collection.delete_many({"user_id": user_id})
        return result.deleted_count

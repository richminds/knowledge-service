"""Tests for the RAG generation layer: which prompt goes to the gateway, and
how an answer that did not come from the knowledge base is labelled."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from rag import llm as llm_mod
from rag.config import settings
from rag.llm import (
    GENERAL_KNOWLEDGE_MARKER,
    answers_without_context,
    build_augmented_prompt,
    build_generation_prompt,
    generate_answer,
)

CHUNKS = [
    Document(page_content="Rule 1 text.", metadata={"title": "Handbook", "source": "hb.pdf"}),
    Document(page_content="Rule 2 text.", metadata={"title": "Handbook", "source": "hb.pdf"}),
]
QUESTION = "What is the capital of India?"


@dataclass
class _Reply:
    content: str


class _FakeClient:
    """Records the messages it was given and returns a canned reply."""

    def __init__(self, content: str = "New Delhi.") -> None:
        self.content = content
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return _Reply(self.content)


@pytest.fixture()
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("rag.llm_gateway_client.get_llm_client", lambda: client)
    return client


# ─────────────────────────────────── prompt selection


def test_grounded_prompt_carries_question_and_numbered_context():
    prompt = build_augmented_prompt(QUESTION, CHUNKS, general_knowledge_fallback=True)
    assert QUESTION in prompt
    assert "[1] title=Handbook" in prompt and "Rule 1 text." in prompt
    assert "[2] title=Handbook" in prompt and "Rule 2 text." in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_fallback_prompt_permits_a_labelled_general_knowledge_answer():
    prompt = build_augmented_prompt(QUESTION, CHUNKS, general_knowledge_fallback=True)
    assert GENERAL_KNOWLEDGE_MARKER in prompt
    assert "general knowledge" in prompt
    assert "using only the context" not in prompt


def test_strict_prompt_refuses_instead():
    prompt = build_augmented_prompt(QUESTION, CHUNKS, general_knowledge_fallback=False)
    assert "using only the context below" in prompt
    assert "say what evidence is missing" in prompt
    assert GENERAL_KNOWLEDGE_MARKER not in prompt


def test_prompt_mode_defaults_to_the_setting(monkeypatch):
    monkeypatch.setattr(settings, "general_knowledge_fallback", False)
    assert "using only the context below" in build_augmented_prompt(QUESTION, CHUNKS)
    monkeypatch.setattr(settings, "general_knowledge_fallback", True)
    assert GENERAL_KNOWLEDGE_MARKER in build_augmented_prompt(QUESTION, CHUNKS)


def test_nothing_selected_sends_the_bare_question(monkeypatch):
    monkeypatch.setattr(settings, "general_knowledge_fallback", True)
    assert answers_without_context([]) is True
    assert build_generation_prompt(QUESTION, []) == QUESTION


def test_nothing_selected_stays_grounded_in_strict_mode(monkeypatch):
    monkeypatch.setattr(settings, "general_knowledge_fallback", False)
    assert answers_without_context([]) is False
    prompt = build_generation_prompt(QUESTION, [])
    assert prompt != QUESTION
    assert "Retrieved context:" in prompt


def test_chunks_selected_always_use_the_grounded_prompt(monkeypatch):
    monkeypatch.setattr(settings, "general_knowledge_fallback", True)
    assert answers_without_context(CHUNKS) is False
    assert "Retrieved context:" in build_generation_prompt(QUESTION, CHUNKS)


# ─────────────────────────────────── generate_answer


async def test_generate_without_context_sends_bare_question_and_labels_the_answer(
    monkeypatch, fake_client
):
    monkeypatch.setattr(settings, "general_knowledge_fallback", True)
    answer = await generate_answer(QUESTION, [])

    sent = fake_client.calls[0]["messages"][0]["content"]
    assert sent == QUESTION  # no grounding wrapper → gateway classifies it normally
    assert fake_client.calls[0]["caller"] == "knowledge-service.generate"
    assert answer == f"{GENERAL_KNOWLEDGE_MARKER}\n\nNew Delhi."


async def test_generate_with_context_returns_the_model_answer_unlabelled(
    monkeypatch, fake_client
):
    monkeypatch.setattr(settings, "general_knowledge_fallback", True)
    fake_client.content = "Rule 1 applies [1]."
    answer = await generate_answer(QUESTION, CHUNKS)

    sent = fake_client.calls[0]["messages"][0]["content"]
    assert "Retrieved context:" in sent and "Rule 1 text." in sent
    assert answer == "Rule 1 applies [1]."  # labelling, if any, is the model's


async def test_generate_without_context_in_strict_mode_is_not_labelled(
    monkeypatch, fake_client
):
    monkeypatch.setattr(settings, "general_knowledge_fallback", False)
    fake_client.content = "The context does not contain this."
    answer = await generate_answer(QUESTION, [])

    assert "Retrieved context:" in fake_client.calls[0]["messages"][0]["content"]
    assert answer == "The context does not contain this."


async def test_gateway_failure_yields_the_unavailability_message(monkeypatch):
    class _Broken:
        async def chat(self, **_kw):
            raise ConnectionError("gateway down")

    monkeypatch.setattr("rag.llm_gateway_client.get_llm_client", lambda: _Broken())
    answer = await generate_answer(QUESTION, [])
    assert answer == llm_mod._UNAVAILABLE_ANSWER


# ─────────────────────────────────── web search key


def test_web_search_key_comes_from_settings(monkeypatch):
    from rag.web_search import WebSearchStore

    monkeypatch.setattr(settings, "tavily_api_key", "tvly-from-settings")
    assert WebSearchStore()._api_key == "tvly-from-settings"
    assert WebSearchStore(api_key="explicit")._api_key == "explicit"


async def test_web_search_without_a_key_makes_no_request(monkeypatch):
    import httpx

    from rag.web_search import WebSearchStore

    def _boom(*_a, **_k):
        raise AssertionError("no HTTP call expected without a key")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    assert await WebSearchStore(api_key="").retrieve("anything") == []


# ─────────────────────────────────── connection pool default


def test_mongo_pool_is_not_pre_warmed_by_default():
    # minPoolSize > 0 makes pymongo open sockets from a background thread; on a
    # frozen/thawed serverless function those handshakes time out and log an
    # error on every wake-up. The pool must grow only on the request path.
    from rag.config import RAGSettings

    assert RAGSettings(_env_file=None).mongo_min_pool_size == 0

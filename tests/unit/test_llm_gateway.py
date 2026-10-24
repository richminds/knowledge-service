"""LLM gateway clients: what ChatClient / EmbeddingClient send, and whose credential goes out.

RemoteLLMClient is a fake (or runs over httpx.MockTransport); nothing leaves the process."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tenacity import wait_none

from app.core.caller_context import CallerTokenMiddleware, caller_token_scope, get_caller_token
from app.core.exceptions import EmbeddingError, GenerationError
from app.rag.generation.llm_client import ChatClient
from app.rag.generation.llm_gateway import _CallerCredential, extract_json_object
from app.rag.generation.llm_gateway_sdk import GatewayError, LLMResponse, StreamEvent
from app.rag.ingestion.embeddings import EmbeddingClient


class FakeRemote:
    """Records RemoteLLMClient calls and replies with canned content / vectors / errors."""

    def __init__(self, content="", vectors=None, errors=()):
        self.content = content
        self.vectors = vectors or [[0.1, 0.2, 0.3]]
        self.errors = list(errors)
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(self.content, "gw-model", 10, 5, "stop")

    async def stream(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        for token in ("Hel", "lo"):
            yield StreamEvent(content=token)
        yield StreamEvent(is_final=True, model="gw-model", prompt_tokens=3, completion_tokens=2)

    async def embed(self, texts, input_type="document", caller="unknown"):
        self.calls.append({"texts": texts, "input_type": input_type, "caller": caller})
        if self.errors:
            raise self.errors.pop(0)
        return [self.vectors[0] for _ in texts]


# -- ChatClient ---------------------------------------------------------------------------------
async def test_complete_json_sends_category_and_caller_and_parses_fenced_json():
    remote = FakeRemote('Here you go:\n```json\n{"route": "graph"}\n```')
    client = ChatClient(remote, caller="ks")

    assert await client.complete_json("sys", "user", purpose="routing") == {"route": "graph"}
    sent = remote.calls[0]
    assert sent["caller"] == "ks:routing" and sent["category"] == "factual_qa"
    assert sent["temperature"] == 0 and sent["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


async def test_complete_json_wraps_unparseable_replies():
    with pytest.raises(GenerationError, match="routing"):
        await ChatClient(FakeRemote("no json here")).complete_json("s", "u", purpose="routing")


async def test_answers_carry_history_and_stream_tokens():
    remote = FakeRemote("25 days")
    client = ChatClient(remote, temperature=0.2, max_tokens=256)
    history = [("user", "hi"), ("assistant", "hello")]

    assert await client.generate_answer("leave?", [], history) == "25 days"
    assert [t async for t in client.stream_answer("leave?", [])] == ["Hel", "lo"]

    sent = remote.calls[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert (sent["caller"], sent["temperature"], sent["max_tokens"]) == (
        "knowledge-service:answer",
        0.2,
        256,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Sure! {"a": {"b": 2}} Hope that helps.', {"a": {"b": 2}}),
    ],
)
def test_extract_json_object_tolerates_wrapping(text, expected):
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "no braces", "[1, 2]", "{not json}"])
def test_extract_json_object_rejects_non_objects(text):
    with pytest.raises(ValueError):
        extract_json_object(text)


# -- EmbeddingClient ----------------------------------------------------------------------------
async def test_embeddings_are_batched_and_labelled():
    remote = FakeRemote(vectors=[[0.0, 1.0]])
    client = EmbeddingClient(remote, batch_size=2, dimensions=2, caller="ks")

    vectors = await client.embed_batch(["a", "b", "c"])
    query = await client.embed_text("q")

    assert len(vectors) == 3 and query == [0.0, 1.0]
    assert [c["texts"] for c in remote.calls] == [["a", "b"], ["c"], ["q"]]
    assert remote.calls[-1]["input_type"] == "query" and remote.calls[0]["caller"] == "ks:embedding"


async def test_embedding_dimension_mismatch_fails_loudly():
    client = EmbeddingClient(FakeRemote(vectors=[[0.1, 0.2, 0.3]]), dimensions=1536)
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIMENSIONS=1536"):
        await client.embed_batch(["a"])


async def test_embeddings_retry_transient_errors_only(monkeypatch):
    monkeypatch.setattr(EmbeddingClient._embed_once.retry, "wait", wait_none())
    throttled = FakeRemote(errors=[GatewayError(429, "rate_limited", "slow down")])
    assert await EmbeddingClient(throttled).embed_batch(["a"]) == [[0.1, 0.2, 0.3]]
    assert len(throttled.calls) == 2

    forbidden = FakeRemote(errors=[GatewayError(403, "forbidden", "no")])
    with pytest.raises(EmbeddingError, match="403"):
        await EmbeddingClient(forbidden).embed_batch(["a"])
    assert len(forbidden.calls) == 1  # 4xx fails fast


# -- caller credential --------------------------------------------------------------------------
async def _headers_sent(service_key: str = "") -> httpx.Headers:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, auth=_CallerCredential(service_key)) as c:
        await c.post("http://llm-gateway.test/v1/llm", json={})
    return seen[0]


async def test_caller_token_is_forwarded_in_preference_to_the_service_key():
    with caller_token_scope("user-token"):
        headers = await _headers_sent("svc-key")
    assert headers["authorization"] == "Bearer user-token" and "x-api-key" not in headers


async def test_service_key_is_used_when_no_caller_is_bound():
    headers = await _headers_sent("svc-key")
    assert headers["x-api-key"] == "svc-key" and "authorization" not in headers

    bare = await _headers_sent()
    assert "x-api-key" not in bare and "authorization" not in bare


def test_middleware_binds_the_bearer_token_for_one_request_only():
    app = FastAPI()
    app.add_middleware(CallerTokenMiddleware)

    @app.get("/token")
    async def token() -> dict:
        return {"token": get_caller_token()}

    client = TestClient(app)
    assert client.get("/token", headers={"Authorization": "Bearer abc"}).json() == {"token": "abc"}
    assert client.get("/token", headers={"Authorization": "Basic abc"}).json() == {"token": ""}
    assert client.get("/token").json() == {"token": ""}
    assert get_caller_token() == ""

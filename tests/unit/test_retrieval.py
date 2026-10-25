from app.models.schemas import ChatMessage, ChatRequest
from app.rag.generation.prompts import NO_CONTEXT_ANSWER, build_context_prompt
from app.rag.retrieval.reranker import ScoreReranker
from app.rag.vectorstore.base import VectorSearchResult


def _result(chunk_id: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(chunk_id=chunk_id, content="c", score=score, source="s")


async def test_reranker_sorts_truncates_and_applies_min_score():
    results = [_result("a", 0.2), _result("b", 0.9), _result("c", 0.5), _result("d", 0.05)]

    ranked = await ScoreReranker(min_score=0.1).rerank("q", results, top_n=2)

    assert [r.chunk_id for r in ranked] == ["b", "c"]


def test_context_prompt_numbers_passages():
    prompt = build_context_prompt("q?", [_result("a", 1.0), _result("b", 0.5)])

    assert "[1] (source: s)" in prompt
    assert "[2] (source: s)" in prompt


async def test_chat_returns_numbered_citations_and_passes_history(
    ingestion_pipeline, retrieval_service, chat_client
):
    await ingestion_pipeline.run("docs", "default", metadata={"account_id": "acct-test"})
    history = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]

    response = await retrieval_service.chat(
        ChatRequest(query="annual leave days", conversation_id="conv-1", history=history), top_k=5
    )

    assert response.citations[0].index == 1
    assert response.citations[0].source.endswith("leave.md")
    assert chat_client.calls[0]["history"] == [("user", "hi"), ("assistant", "hello")]
    assert response.conversation_id == "conv-1"


async def test_chat_without_context_skips_llm(retrieval_service, chat_client):
    response = await retrieval_service.chat(
        ChatRequest(query="anything", collection="empty-collection"), top_k=5
    )

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.citations == []
    assert chat_client.calls == []

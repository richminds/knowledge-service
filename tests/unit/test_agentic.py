"""Phase 5 (Part D): tools, planning, control limits, verification, UX hooks."""

import asyncio

import pytest

from app.core.config import RagMode
from app.models.schemas import ChatRequest
from app.rag.agents.tools import (
    Tool,
    calculator_tool,
    date_math_tool,
    fetch_document_tool,
    keyword_tool,
    safe_eval,
)
from app.rag.ingestion.pipeline import LoadedDocument
from app.rag.retrieval.retriever import VectorRetriever
from app.rag.strategies.agentic import AgenticRagStrategy, AgentLimits, build_checkpointer
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult

DEFAULT = RetrievalFilters(collection="default")


def _hit(cid: str, text: str) -> VectorSearchResult:
    return VectorSearchResult(chunk_id=cid, content=text, score=1.0, source=f"{cid}.md")


def _static_tool(
    name: str,
    results: list[VectorSearchResult],
    calls: list | None = None,
    delay: float = 0,
    fail: bool = False,
) -> Tool:
    async def run(args, filters, top_k):
        if calls is not None:
            calls.append((name, args.get("query")))
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("tool down")
        return list(results)

    return Tool(name, f"{name} tool", {"query": {"type": "string"}}, run, {"vector"})


# -- tools (D2) -------------------------------------------------------------------------------
async def test_utility_tools():
    calc = await calculator_tool().run({"expression": "(120*5)/2"}, DEFAULT, 1)
    assert calc[0].content == "(120*5)/2 = 300"
    blocked = await calculator_tool().run(
        {"expression": "__import__('os').system('x')"}, DEFAULT, 1
    )
    assert "Could not evaluate" in blocked[0].content
    days = await date_math_tool().run({"date": "2024-01-01", "end_date": "2024-03-01"}, DEFAULT, 1)
    assert days[0].content.endswith("60 days")
    with pytest.raises(ValueError):
        safe_eval("2 ** 10000")


async def test_search_tools_enforce_acl(ingestion_pipeline, vector_store, embedding_client):
    await ingestion_pipeline.run_for_documents(
        [LoadedDocument("secret.md", "Board salary bands for SECRET-42", acl=["grp-board"])],
        "default",
        metadata={"account_id": "acct-test"},
    )
    outsider = RetrievalFilters(collection="default", principals=frozenset({"*"}))
    keyword = keyword_tool(VectorRetriever(embedding_client, vector_store))
    assert await keyword.run({"query": "SECRET-42"}, outsider, 5) == []
    assert await keyword.run({"query": "SECRET-42"}, DEFAULT, 5)
    doc_id = next(iter(vector_store.records.values())).doc_id
    assert await fetch_document_tool(vector_store).run({"doc_id": doc_id}, outsider, 5) == []
    assert (await fetch_document_tool(vector_store).run({"doc_id": doc_id}, DEFAULT, 5))[0].content


# -- workflow ---------------------------------------------------------------------------------
async def test_planner_runs_sub_questions_in_parallel(chat_client):
    calls: list = []
    chat_client.route = {"tools": [{"tool": "docs"}], "complexity": "multi_step"}
    chat_client.agent_plan = ["policy for 2023", "policy for 2024"]
    agent = AgenticRagStrategy(
        chat_client, tools=[_static_tool("docs", [_hit("a", "x"), _hit("b", "y")], calls)]
    )
    outcome = await agent.retrieve("Compare the 2023 and 2024 policy", DEFAULT, 4)
    assert sorted(q for _, q in calls) == ["policy for 2023", "policy for 2024"]
    assert any(s.step == "plan" for s in outcome.trace)


async def test_failing_or_slow_tool_falls_back(chat_client):
    chat_client.route = {"tools": [{"tool": "broken"}]}
    agent = AgenticRagStrategy(
        chat_client,
        tools=[
            _static_tool("broken", [], fail=True),
            _static_tool("backup", [_hit("a", "x"), _hit("b", "y")]),
        ],
        limits=AgentLimits(step_timeout_seconds=0.5),
    )
    outcome = await agent.retrieve("q", DEFAULT, 4)
    retrieve = next(s for s in outcome.trace if s.step == "retrieve")
    assert retrieve.detail["errors"] == {"broken": "RuntimeError"}
    assert {c.chunk_id for c in outcome.contexts} == {"a", "b"}

    chat_client.route = {"tools": [{"tool": "slow"}]}
    slow = AgenticRagStrategy(
        chat_client,
        tools=[
            _static_tool("slow", [_hit("z", "z")], delay=2),
            _static_tool("backup", [_hit("a", "x"), _hit("b", "y")]),
        ],
        limits=AgentLimits(step_timeout_seconds=0.1),
    )
    outcome = await slow.retrieve("q", DEFAULT, 4)
    assert next(s for s in outcome.trace if s.step == "retrieve").detail["errors"] == {
        "slow": "TimeoutError"
    }


async def test_budget_stops_the_loop_and_searches_are_not_repeated(chat_client):
    calls: list = []
    chat_client.grade_keep = set()
    agent = AgenticRagStrategy(
        chat_client,
        tools=[_static_tool("docs", [_hit("a", "x")], calls)],
        limits=AgentLimits(max_iterations=5, max_tool_calls=2),
    )
    outcome = await agent.retrieve("Who runs Finance?", DEFAULT, 4)
    assert len(calls) == 2  # A-34 tool-call cap
    assert len(set(calls)) == len(calls)  # A-38 never the same search twice
    assert outcome.trace[-1].detail.get("budget_stop") == "tool_calls" or any(
        s.step == "sufficiency" and s.detail.get("budget_stop") == "tool_calls"
        for s in outcome.trace
    )


async def test_failed_check_reflects_then_flags_for_review(chat_client):
    chat_client.agent_check = {
        "grounded": False,
        "addresses_question": True,
        "unsupported": ["x"],
        "missing": "budget figures",
    }
    calls: list = []
    agent = AgenticRagStrategy(
        chat_client,
        tools=[_static_tool("docs", [_hit("a", "x"), _hit("b", "y")], calls)],
        limits=AgentLimits(max_iterations=4, max_reflections=1),
    )
    outcome = await agent.retrieve("What is the budget?", DEFAULT, 4)
    assert ("docs", "budget figures") in calls  # A-31: retried with what was missing
    assert [s.step for s in outcome.trace].count("answer_check") == 2
    assert outcome.review_required  # A-44


async def test_clarifying_question(chat_client):
    chat_client.route = {
        "action": "clarify",
        "clarifying_question": "Which year do you mean?",
        "tools": [],
    }
    agent = AgenticRagStrategy(
        chat_client,
        tools=[_static_tool("docs", [_hit("a", "x")])],
        limits=AgentLimits(allow_clarify=True),
    )
    outcome = await agent.retrieve("What was the budget?", DEFAULT, 4)
    assert outcome.clarification == "Which year do you mean?" and outcome.contexts == []


async def test_memory_checkpointer_runs(chat_client):
    agent = AgenticRagStrategy(
        chat_client,
        tools=[_static_tool("docs", [_hit("a", "x"), _hit("b", "y")])],
        checkpointer=build_checkpointer("memory"),
    )
    outcome = await agent.retrieve("q", DEFAULT, 4)
    assert outcome.answer


# -- service integration ------------------------------------------------------------------------
async def test_service_uses_agent_answer_and_clarification(
    retrieval_service, ingestion_pipeline, chat_client
):
    await ingestion_pipeline.run_for_documents(
        [LoadedDocument("org.md", "Alice manages Finance. Finance funds Travel.")],
        "default",
        targets=["vector", "graph"],
        metadata={"account_id": "acct-test"},
    )
    calls_before = len(chat_client.calls)
    response = await retrieval_service.chat(
        ChatRequest(query="Who runs Finance?", rag_mode=RagMode.AGENTIC)
    )
    assert response.answer_type == "answer" and response.rag_mode == RagMode.AGENTIC
    assert len(chat_client.calls) == calls_before + 1  # generated once, inside the agent
    assert {c.retriever for c in response.citations} >= {"vector"}

"""Agent tools (D2). Each tool has a name, a description that tells the router WHEN to use
it (A-16), a JSON parameter schema, and receives the caller's RetrievalFilters so the
tool - not the agent - enforces permissions (A-17). All tools are read-only (A-40).

  semantic_search      wraps the semantic strategy (A-07)
  keyword_search       BM25 for exact ids, codes, names (A-08)
  graph_local_search   entities, relations, multi-hop (A-09)
  graph_global_search  community reports for corpus-wide questions (A-10)
  graph_query          templated / validated Cypher (A-11)
  fetch_document       the full parent document of a passage (A-14)
  calculator, date_math  utility tools (A-15)
Not included (TODO): sql_query (A-12) and sandboxed code execution (A-15, A-40).
No external web search (A-13): answers come from the indexed corpus only.
 MCP exposure of these tools (A-18) is a future option.
"""

import ast
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any

from app.rag.graph.strategy_modes import run_graph_mode
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult

ToolFn = Callable[[dict[str, Any], RetrievalFilters, int], Awaitable[list[VectorSearchResult]]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: ToolFn
    tags: set[str] = field(default_factory=set)

    def spec(self) -> str:
        params = ", ".join(
            f"{k}: {v.get('description', v.get('type', ''))}" for k, v in self.parameters.items()
        )
        return f"- {self.name}({params}): {self.description}"


QUERY_PARAM = {"query": {"type": "string", "description": "search query"}}


def _tag(results: list[VectorSearchResult], tool: str) -> list[VectorSearchResult]:
    for r in results:
        r.metadata.setdefault("retriever", r.metadata.get("retriever", tool))
        r.metadata["tool"] = tool
    return results


def strategy_tool(name: str, description: str, strategy, tags: set[str]) -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        outcome = await strategy.retrieve(str(args.get("query", "")), filters, top_k)
        return _tag(outcome.contexts, name)

    return Tool(name, description, QUERY_PARAM, run, tags)


def keyword_tool(retriever) -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        return _tag(
            await retriever.keyword_search(str(args.get("query", "")), filters, top_k),
            "keyword_search",
        )

    return Tool(
        "keyword_search",
        "Exact-term search (BM25). Use for document numbers, codes, acronyms, product or person "
        "names.",
        QUERY_PARAM,
        run,
        {"vector"},
    )


def graph_mode_tool(name: str, description: str, graph_strategy, mode: str) -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        return _tag(
            await run_graph_mode(graph_strategy, mode, str(args.get("query", "")), filters, top_k),
            name,
        )

    return Tool(name, description, QUERY_PARAM, run, {"graph"})


def fetch_document_tool(vector_store) -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        doc_id = str(args.get("doc_id", ""))
        chunks = await vector_store.get_document_chunks(doc_id, filters, limit=60) if doc_id else []
        if not chunks:
            return []
        first = chunks[0]
        text = "\n\n".join(c.content for c in chunks)
        return [
            replace(
                first,
                chunk_id=f"doc:{doc_id}",
                content=text[:12000],
                score=1.0,
                metadata={
                    **first.metadata,
                    "retriever": "fetch_document",
                    "tool": "fetch_document",
                },
            )
        ]

    return Tool(
        "fetch_document",
        "Get the full text of a document already found (pass its doc_id) when a passage is not "
        "enough.",
        {"doc_id": {"type": "string", "description": "doc_id of a retrieved passage"}},
        run,
        {"vector"},
    )


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Arithmetic only: numbers and + - * / % ** parentheses (no names, calls or attributes)."""

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    return walk(ast.parse(expression, mode="eval"))


def _note(tool: str, text: str) -> list[VectorSearchResult]:
    return [
        VectorSearchResult(
            chunk_id=f"{tool}:{abs(hash(text)) % 10**8}",
            content=text,
            score=1.0,
            source=tool,
            metadata={"retriever": tool, "tool": tool, "title": tool},
        )
    ]


def calculator_tool() -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        expression = str(args.get("expression", ""))
        try:
            return _note("calculator", f"{expression} = {safe_eval(expression):g}")
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return _note("calculator", f"Could not evaluate {expression!r}: {exc}")

    return Tool(
        "calculator",
        "Arithmetic on numbers found in sources (totals, percentages, differences).",
        {"expression": {"type": "string", "description": "e.g. (120*5)/2"}},
        run,
        {"utility"},
    )


def date_math_tool() -> Tool:
    async def run(args: dict, filters: RetrievalFilters, top_k: int) -> list[VectorSearchResult]:
        try:
            start = date.fromisoformat(str(args.get("date") or date.today().isoformat()))
            end = args.get("end_date")
            if end:
                days = (date.fromisoformat(str(end)) - start).days
                return _note("date_math", f"From {start} to {end}: {days} days")
            shifted = start + timedelta(days=int(args.get("days", 0)))
            return _note("date_math", f"{start} + {int(args.get('days', 0))} days = {shifted}")
        except (ValueError, TypeError) as exc:
            return _note("date_math", f"Invalid date input: {exc}")

    return Tool(
        "date_math",
        "Date arithmetic: days between two dates, or a date plus N days (deadlines).",
        {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD (optional)"},
            "days": {"type": "integer", "description": "days to add (optional)"},
        },
        run,
        {"utility"},
    )

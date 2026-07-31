"""Local CLI for the Knowledge Service RAG pipeline — no HTTP server needed.

Runs the same LangGraph pipelines the API serves, in-process, against the
configured MongoDB and LLM Gateway. Useful for local ingestion/testing
without standing up the FastAPI app. Ported from the source implementation's
``practice-rag`` CLI (``portless/backend/shared/rag/src/cli.py``).

Usage::

    python scripts/rag_cli.py ingest --path ./data --strategy recursive
    python scripts/rag_cli.py ask "What is the refund policy?" --filter section=Refunds
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.graph import build_ingestion_graph, build_query_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Service RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Parse, chunk, embed, and insert documents"
    )
    ingest_parser.add_argument(
        "--path", action="append", default=None, help="File or folder to ingest"
    )
    ingest_parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy",
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Run retrieval, reranking, augmentation, and generation"
    )
    ask_parser.add_argument("question", help="Question to answer from the RAG index")
    ask_parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Exact metadata filter in key=value form. Can be repeated.",
    )
    ask_parser.add_argument("--user", default=None, help="User ID for authorization filtering")
    ask_parser.add_argument("--team", default=None, help="Team ID for authorization filtering")

    args = parser.parse_args()

    if args.command == "ingest":
        if not args.path:
            print("Error: provide at least one --path argument.")
            raise SystemExit(1)
        graph = build_ingestion_graph()
        result = asyncio.run(
            graph.ainvoke({"input_paths": args.path, "chunk_strategy": args.strategy})
        )
        print(f"Inserted vector chunks: {result.get('inserted_count', 0)}")
        print(f"Inserted graph chunks: {result.get('graph_inserted_count', 0)}")
        return

    if args.command == "ask":
        graph = build_query_graph()
        result = asyncio.run(
            graph.ainvoke(
                {
                    "question": args.question,
                    "metadata_filter": parse_metadata_filters(args.filter or []),
                    "user_id": args.user,
                    "team_id": args.team,
                }
            )
        )
        print(result["answer"])
        print("\nCitation validation:", result.get("citation_validation"))
        print("Evaluation metrics:", result.get("evaluation_metrics"))


def parse_metadata_filters(filters: list[str]) -> dict[str, Any]:
    """Parse CLI metadata filters from `key=value` strings."""
    parsed: dict[str, Any] = {}
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Invalid filter '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        parsed[key.strip()] = _coerce_filter_value(value.strip())
    return parsed


def _coerce_filter_value(value: str) -> Any:
    """Convert a CLI filter value into a simple metadata scalar."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    main()

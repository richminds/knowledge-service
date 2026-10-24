"""Per-request progress channel (A-42): retrieval/agent steps are pushed here as they finish,
so the streaming endpoint can show "Searching policies...", "Checking graph..." live."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

ProgressCallback = Callable[[str, dict[str, Any]], None]

_callback: ContextVar[ProgressCallback | None] = ContextVar("progress_callback", default=None)

# Human-readable labels for streamed steps.
STEP_LABELS = {
    "query_analysis": "Understanding the question",
    "route": "Choosing where to look",
    "plan": "Planning sub-questions",
    "retrieve": "Searching",
    "vector_search": "Searching documents",
    "graph_traversal": "Checking the knowledge graph",
    "community_reports": "Reviewing topic summaries",
    "grade": "Checking relevance of what was found",
    "rewrite": "Rephrasing the search",
    "sufficiency": "Checking whether there is enough evidence",
    "generate": "Drafting the answer",
    "hallucination_check": "Verifying the answer against sources",
    "relevance_check": "Checking the answer addresses the question",
}


@contextmanager
def progress(callback: ProgressCallback):
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def emit(step: str, detail: dict[str, Any]) -> None:
    callback = _callback.get()
    if callback is not None:
        try:
            callback(step, detail)
        except Exception:  # noqa: BLE001 - progress must never break retrieval
            pass

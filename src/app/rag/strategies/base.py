"""Common contract for the RAG strategies (semantic / graph / agentic).

Each strategy only decides *which contexts* to ground the answer on; answer generation
(sync and streaming) is shared in RetrievalService, so strategies are compared on equal
terms when validating them side by side.
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app.core import metrics, progress
from app.core.config import RagMode
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult

# (role, content) pairs of prior turns, oldest first.
History = Sequence[tuple[str, str]]


@dataclass
class TraceStep:
    step: str
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class RetrievalOutcome:
    contexts: list[VectorSearchResult]
    trace: list[TraceStep] = field(default_factory=list)
    # Set by strategies that already produced and verified an answer (agentic, A-28..A-31);
    # the service then uses it instead of generating again.
    answer: str | None = None
    # Set when the strategy needs the user to clarify the question first (A-45).
    clarification: str | None = None
    # Set when the answer should be reviewed by a person before relying on it (A-44).
    review_required: bool = False


def _span_value(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, default=str)[:1000]


@contextmanager
def traced(trace: list[TraceStep], step: str, **detail: Any) -> Iterator[dict[str, Any]]:
    """Time a step: append it to the response `trace`, emit an OpenTelemetry span
    (`rag.<step>`, nested under the request in App Insights) and a step-latency metric.
    Callers may add to the yielded detail dict; it becomes span attributes too."""
    start = time.perf_counter()
    info: dict[str, Any] = dict(detail)
    with metrics.tracer.start_as_current_span(f"rag.{step}") as span:
        try:
            yield info
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            trace.append(TraceStep(step, info, duration_ms))
            for key, value in info.items():
                span.set_attribute(f"rag.{key}", _span_value(value))
            metrics.step_duration.record(duration_ms, {"rag.step": step})
            progress.emit(step, {**info, "duration_ms": duration_ms})


class RagStrategy(ABC):
    mode: RagMode

    @abstractmethod
    async def retrieve(
        self, query: str, filters: RetrievalFilters, top_k: int, history: History = ()
    ) -> RetrievalOutcome:
        """Return the contexts the answer should be grounded on (C-023 retriever contract).

        `filters` carries the collection, the caller's ACL principals and any metadata
        filters; implementations must apply them in the underlying store.
        """

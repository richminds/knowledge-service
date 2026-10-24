"""Per-request LLM token accounting.

LLM clients call `record_usage(...)` after every call. Inside `track_usage()` the counts
also accumulate into a tracker for the current request (a context variable, so concurrent
requests don't mix), which the API returns as `usage` and logs with the answer.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.core import metrics


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    by_purpose: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_current: ContextVar[UsageTracker | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage(tracker: UsageTracker | None = None) -> Iterator[UsageTracker]:
    """Accumulate usage into `tracker` (a new one if omitted) for the enclosed calls.

    Pass the same tracker again to keep counting across phases that run in different
    contexts, e.g. retrieval in the endpoint and token streaming in the response body.
    """
    tracker = tracker or UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def record_usage(purpose: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    attrs = {"llm.purpose": purpose, "llm.model": model}
    metrics.llm_tokens.add(prompt_tokens, {**attrs, "token.type": "prompt"})
    metrics.llm_tokens.add(completion_tokens, {**attrs, "token.type": "completion"})
    tracker = _current.get()
    if tracker is not None:
        tracker.prompt_tokens += prompt_tokens
        tracker.completion_tokens += completion_tokens
        tracker.llm_calls += 1
        tokens = prompt_tokens + completion_tokens
        tracker.by_purpose[purpose] = tracker.by_purpose.get(purpose, 0) + tokens
        tracker.by_model[model] = tracker.by_model.get(model, 0) + tokens


def current_usage() -> UsageTracker | None:
    """The tracker of the request being served (for per-run budgets, A-35)."""
    return _current.get()

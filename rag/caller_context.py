"""The calling user's credential for the current logical request.

Every LLM and embedding call this service makes goes out through the API
Gateway, which authenticates it like any other traffic (see
``rag/llm_gateway_client.py``). So the outbound call needs the *caller's* token,
not a service credential — that is what makes the gateway the single place
where per-user rate limits, budgets and metering are enforced, instead of RAG
traffic arriving as one anonymous service and bypassing all three.

Getting the token to the call site is the awkward part. It arrives on an HTTP
request; it is needed in ``rag/llm.py``, ``rag/embeddings.py``,
``rag/security.py`` and ``rag/evaluation.py``, none of which take a request and
several of which are LangChain callbacks with signatures this service does not
control. Threading a token parameter through every layer between would touch
most of the package to deliver one value.

A ``contextvars.ContextVar`` carries it ambiently instead — the same mechanism
and the same rationale as ``rag/log_context.py``, which already does this for
the correlation ID. Everything the request awaits sees it, with no parameter
threading.

**This one holds a live credential**, which the correlation ID does not, so two
rules apply that do not apply there:

  * **Always reset it.** A token left bound outlives its request and would be
    attached to whatever ran next on that context. Bind through
    ``caller_token_scope`` unless you have a reason not to.
  * **Never log it.** There is no accessor here that formats it, and none
    should be added. ``has_caller_token`` exists so callers can branch on
    presence without ever handling the value.

**Background work must be handed the token explicitly.** FastAPI's
``BackgroundTasks`` run after the response has travelled back up the middleware
stack, by which point the middleware's ``finally`` has already reset this var.
Ingestion therefore passes the token into the job and re-binds it there — see
``app/services/ingest_service.py``. It is a snapshot: a job outliving the
token's expiry will start failing its embedding calls, which is a real limit of
routing everything through the gateway and is documented in that module.
"""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

_caller_token_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "knowledge_service_caller_token", default=""
)


def get_caller_token() -> str:
    """The bearer token of whoever is being served right now, or "".

    Empty is normal and not an error: the CLI (``scripts/rag_cli.py``), the
    test suite and any in-process caller have no HTTP request and so no token.
    ``rag/llm_gateway_client.py`` decides what to do about that.
    """
    return _caller_token_var.get()


def has_caller_token() -> bool:
    """Whether a token is bound, without handing the value to the caller.

    Use this for logging and branching. The value itself should only ever be
    read by the code that puts it on the wire.
    """
    return bool(_caller_token_var.get())


def bind_caller_token(token: str) -> contextvars.Token:
    """Bind a caller's token for the current context.

    Returns a token for ``reset_caller_token``. Prefer ``caller_token_scope``
    — a missed reset here leaks a credential into an unrelated request.
    """
    return _caller_token_var.set(token or "")


def reset_caller_token(token: contextvars.Token) -> None:
    _caller_token_var.reset(token)


@contextmanager
def caller_token_scope(token: str) -> Iterator[None]:
    """Bind a caller's token for the duration of the ``with`` block."""
    var_token = bind_caller_token(token)
    try:
        yield
    finally:
        reset_caller_token(var_token)

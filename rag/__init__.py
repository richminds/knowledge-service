"""Knowledge Service RAG core — MongoDB Atlas vector store + LLM Gateway generation.

This package is the portable, framework-free core extracted from
``portless/backend/shared/rag``. It holds no FastAPI dependency and no
provider SDK — every LLM and embedding call goes out over HTTP to an LLM
Gateway deployment (see ``rag/llm_gateway_client.py``). ``app/`` is the thin
HTTP layer wired on top of it, exactly mirroring the ``features/`` + ``app/``
split in the LLM Gateway project this service was built alongside.

The public names (``query``, ``ingest``, the graph builders, ``settings``) are
loaded lazily via PEP 562 ``__getattr__``, so importing this package — or any
single submodule such as a graph-store backend — stays cheap: the heavy
pipelines in ``ingestion.py`` / ``retrieval.py`` (and their optional
dependencies) are only imported the first time one of those names is
actually used. Ingestion and retrieval are separate modules — see
``ingestion.py``'s docstring for why — so importing one never pulls in the
other's dependencies either.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "build_ingestion_graph",
    "build_query_graph",
    "ingest",
    "query",
    "settings",
]

# name → relative module that defines it
_LAZY_EXPORTS = {
    "build_ingestion_graph": ".ingestion",
    "ingest": ".ingestion",
    "build_query_graph": ".retrieval",
    "query": ".retrieval",
    "settings": ".config",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # import-time only for type checkers, never at runtime
    from .config import settings
    from .ingestion import build_ingestion_graph, ingest
    from .retrieval import build_query_graph, query

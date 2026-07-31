"""Document chunking — recursive and semantic strategies.

Returns (child_chunks, parent_docs) so the caller can persist parent
documents asynchronously to MongoDB without blocking the CPU-bound
chunking logic on I/O.
"""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .llm_gateway_sdk import GatewayEmbeddingsClient

ChunkStrategy = Literal["recursive", "semantic"]


def chunk_documents(
    documents: list[Document],
    embeddings: GatewayEmbeddingsClient,
    strategy: ChunkStrategy = "recursive",
) -> tuple[list[Document], dict[str, Document]]:
    """Split documents into child chunks and produce parent context groups.

    Returns:
        (child_chunks, parent_docs)
        child_chunks — list of LangChain Documents with chunk metadata
        parent_docs  — {parent_id: Document} mapping for async persistence

    The caller is responsible for awaiting parent_store.save(parent_docs).

    This is a plain (non-async) function — the semantic strategy calls
    ``embeddings.embed_documents`` synchronously (one blocking HTTP round
    trip to the LLM Gateway). Callers on the async path (see ``ingestion.py``'s
    ``_chunk`` node) run this off the event loop via ``run_in_threadpool``.
    """
    if strategy == "semantic":
        return semantic_chunk_documents(documents, embeddings)
    return recursive_chunk_documents(documents)


def recursive_chunk_documents(
    documents: list[Document],
) -> tuple[list[Document], dict[str, Document]]:
    """Split documents with LangChain's recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return _number_chunks(chunks, "recursive")


def semantic_chunk_documents(
    documents: list[Document],
    embeddings: GatewayEmbeddingsClient,
) -> tuple[list[Document], dict[str, Document]]:
    """Split documents by detecting semantic topic shifts via embedding similarity."""
    chunks: list[Document] = []
    for document in documents:
        units = _split_semantic_units(document.page_content)
        if len(units) <= 1:
            chunks.append(
                Document(
                    page_content=document.page_content,
                    metadata={**document.metadata, "chunk_strategy": "semantic"},
                )
            )
            continue

        vectors = embeddings.embed_documents(units)
        similarities = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = float(np.percentile(similarities, settings.semantic_breakpoint_percentile))

        current: list[str] = [units[0]]
        for idx, similarity in enumerate(similarities, start=1):
            current_text = "\n\n".join(current)
            should_break = similarity < threshold and len(current_text) >= settings.chunk_size // 3
            too_large = len(current_text) >= settings.chunk_size
            if should_break or too_large:
                chunks.append(
                    Document(
                        page_content=current_text,
                        metadata={**document.metadata, "chunk_strategy": "semantic"},
                    )
                )
                current = [units[idx]]
            else:
                current.append(units[idx])

        if current:
            chunks.append(
                Document(
                    page_content="\n\n".join(current),
                    metadata={**document.metadata, "chunk_strategy": "semantic"},
                )
            )

    return _number_chunks(chunks, "semantic")


def _split_semantic_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= settings.chunk_size:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    return units


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _number_chunks(
    chunks: list[Document],
    strategy: str,
) -> tuple[list[Document], dict[str, Document]]:
    """Attach chunk metadata and build parent context groups.

    Returns (numbered_child_chunks, parent_documents).
    Parent documents are NOT persisted here — the graph ingestion node
    persists them asynchronously to MongoDB after this function returns.
    """
    total_by_group: dict[str, int] = {}
    grouped_chunks: dict[str, list[tuple[int, Document]]] = {}

    for chunk in chunks:
        group_key = _chunk_group_key(chunk)
        chunk_index = total_by_group.get(group_key, 0)
        total_by_group[group_key] = chunk_index + 1
        grouped_chunks.setdefault(group_key, []).append((chunk_index, chunk))

    numbered: list[Document] = []
    parent_documents: dict[str, Document] = {}

    for group_key, group_items in grouped_chunks.items():
        parent_groups = _build_parent_groups(group_items)
        for chunk_index, chunk in group_items:
            parent_index = chunk_index // settings.child_chunks_per_parent
            parent_id = f"{group_key}::{strategy}::parent::{parent_index}"
            if parent_id not in parent_documents:
                parent_documents[parent_id] = Document(
                    page_content=parent_groups[parent_index],
                    metadata={
                        **chunk.metadata,
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "parent_chunk_count": settings.child_chunks_per_parent,
                    },
                )
            numbered.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": f"{group_key}::{strategy}::child::{chunk_index}",
                        "chunk_index": chunk_index,
                        "chunk_strategy": chunk.metadata.get("chunk_strategy", strategy),
                        "chunk_chars": len(chunk.page_content),
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                    },
                )
            )

    return numbered, parent_documents


def _chunk_group_key(chunk: Document) -> str:
    source = str(chunk.metadata.get("source", "unknown"))
    section = str(chunk.metadata.get("section", "unknown-section"))
    page = str(chunk.metadata.get("page_number", "no-page"))
    safe_section = re.sub(r"[^a-zA-Z0-9_.-]+", "-", section).strip("-") or "section"
    return f"{source}::{safe_section}::page-{page}"


def _build_parent_groups(group_items: list[tuple[int, Document]]) -> dict[int, str]:
    parent_groups: dict[int, list[str]] = {}
    for chunk_index, chunk in group_items:
        parent_index = chunk_index // settings.child_chunks_per_parent
        parent_groups.setdefault(parent_index, []).append(chunk.page_content)
    return {idx: "\n\n".join(parts) for idx, parts in parent_groups.items()}

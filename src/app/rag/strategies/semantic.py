"""Semantic and hybrid RAG (Part B) over the shared index. One strategy, two modes:

  semantic  dense vector search only
  hybrid    dense vector + BM25 keyword search, fused with RRF by the store (S-18..S-22)

  query transform (none | multi_query | hyde | step_back)       S-23, S-24, C-164
  search per query (+ glossary synonyms for BM25 in hybrid)      S-18..S-22, S-08
  RRF fusion of the per-query lists                              S-22, C-168
  multi-vector mapping: question/summary hit -> its chunk        S-25
  similarity threshold                                           S-27
  empty-result fallback: relax filters, then step-back query     S-32
  recency boost on effective_date                                S-31
  MMR diversification                                            S-28
  parent-document / sentence-window expansion                    S-29, S-30
  hierarchical: best documents (via their summaries) first, then  S-26, S-14
  chunks within them
Reranking, token budgeting and ordering happen afterwards in the shared ContextAssembler.
"""

import asyncio
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from app.core.config import RagMode
from app.rag.context.fusion import mmr, rrf_fuse, score_cutoff
from app.rag.query.glossary import Glossary
from app.rag.query.transforms import QueryTransformer
from app.rag.retrieval.retriever import VectorRetriever
from app.rag.strategies.base import History, RagStrategy, RetrievalOutcome, traced
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult, VectorStore


@dataclass
class SemanticOptions:
    query_transform: str = "none"  # none | multi_query | hyde | step_back
    multi_query_count: int = 3
    min_score: float = 0.0
    mmr: bool = False
    mmr_lambda: float = 0.7
    expand: str = "none"  # none | parent | window
    window: int = 1
    recency_boost: float = 0.0  # extra weight for the newest documents; 0 = off
    recency_half_life_days: int = 365
    fallback: bool = True
    hierarchical: bool = False
    hierarchical_docs: int = 5


def recency_factor(effective_date: str | None, boost: float, half_life_days: int) -> float:
    if not boost or not effective_date:
        return 1.0
    try:
        when = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    age_days = max(0.0, (datetime.now(UTC) - when).total_seconds() / 86400)
    return 1.0 + boost * math.pow(0.5, age_days / max(1, half_life_days))


class SemanticRagStrategy(RagStrategy):
    def __init__(
        self,
        retriever: VectorRetriever,
        vector_store: VectorStore,
        transformer: QueryTransformer | None = None,
        glossary: Glossary | None = None,
        options: SemanticOptions | None = None,
        mode: RagMode = RagMode.SEMANTIC,
    ) -> None:
        if mode not in (RagMode.SEMANTIC, RagMode.HYBRID):
            raise ValueError(f"SemanticRagStrategy serves semantic or hybrid, not {mode}")
        self.mode = mode
        # hybrid: the keyword side always runs, whatever RETRIEVAL_MODE says.
        self._keyword = mode == RagMode.HYBRID
        self._retriever = retriever
        self._store = vector_store
        self._transformer = transformer
        self._glossary = glossary or Glossary()
        self._options = options or SemanticOptions()

    # -- search building blocks ---------------------------------------------------------------
    def _keyword_text(self, query: str) -> str | None:
        if not self._keyword:
            return None
        variants = self._glossary.keyword_variants(query)
        return f"{query} {' '.join(variants)}" if variants else query

    async def _hybrid(self, query: str, filters: RetrievalFilters, top_k: int) -> list:
        embedding = await self._retriever.embed(query)
        results = await self._store.search(
            embedding,
            filters,
            top_k,
            query_text=self._keyword_text(query),
            hybrid=self._keyword,
        )
        return score_cutoff(results, min_score=self._options.min_score) if results else results

    async def _search(self, query: str, filters: RetrievalFilters, top_k: int, trace) -> list:
        options = self._options
        transform = options.query_transform if self._transformer else "none"
        queries = [query]
        hyde_passage = None
        if transform != "none":
            with traced(trace, "query_transform", kind=transform) as info:
                if transform == "multi_query":
                    queries = await self._transformer.multi_query(query, options.multi_query_count)
                elif transform == "step_back":
                    queries = [query, await self._transformer.step_back(query)]
                elif transform == "hyde":
                    hyde_passage = await self._transformer.hyde(query)
                info["queries"] = queries

        search = "hybrid" if self._keyword else "vector"
        with traced(
            trace, "vector_search", search=search, queries=len(queries), top_k=top_k
        ) as info:
            searches = [self._hybrid(q, filters, top_k) for q in queries]
            if hyde_passage:
                embedding = await self._retriever.embed(hyde_passage)
                searches.append(self._retriever.retrieve_by_vector(embedding, filters, top_k))
            lists = [r for r in await asyncio.gather(*searches) if r]
            results = rrf_fuse(lists) if len(lists) > 1 else (lists[0] if lists else [])
            info["results"] = len(results)
        return results[:top_k]

    async def _map_multi_vector(self, results: list, filters: RetrievalFilters) -> list:
        """S-25: a hit on a generated question/summary stands for its target chunk."""
        targets = {r.target_id for r in results if r.kind != "chunk" and r.target_id}
        if not targets:
            return results
        chunks = {c.chunk_id: c for c in await self._store.get_by_ids(sorted(targets), filters)}
        mapped: dict[str, VectorSearchResult] = {}
        for result in results:
            chunk = chunks.get(result.target_id) if result.kind != "chunk" else result
            if chunk is None:
                continue
            best = mapped.get(chunk.chunk_id)
            if best is None or result.score > best.score:
                mapped[chunk.chunk_id] = replace(chunk, score=result.score)
        return sorted(mapped.values(), key=lambda r: r.score, reverse=True)

    async def _expand(self, results: list, filters: RetrievalFilters) -> list:
        mode = self._options.expand
        if mode == "parent":
            parent_ids = sorted({r.parent_id for r in results if r.parent_id})
            parents = {p.chunk_id: p for p in await self._store.get_by_ids(parent_ids, filters)}
            expanded: dict[str, VectorSearchResult] = {}
            for r in results:
                parent = parents.get(r.parent_id)
                key = parent.chunk_id if parent else r.chunk_id
                if parent is None:
                    expanded.setdefault(key, r)
                    continue
                if key not in expanded:  # siblings collapse into one parent
                    expanded[key] = replace(parent, score=r.score, matched_chunks=[])
                # Keep which chunks matched, so validation views can show them under the parent.
                expanded[key].matched_chunks.append(
                    {"chunk_id": r.chunk_id, "score": r.score, "preview": r.content[:200]}
                )
            return list(expanded.values())
        if mode == "window":
            neighbour_ids = sorted(
                {i for r in results for i in (r.prev_id, r.next_id) if i}
                - {r.chunk_id for r in results}
            )
            neighbours = {
                n.chunk_id: n for n in await self._store.get_by_ids(neighbour_ids, filters)
            }
            out = []
            for r in results:
                before = neighbours.get(r.prev_id)
                after = neighbours.get(r.next_id)
                text = "\n".join(x.content for x in (before, r, after) if x is not None)
                out.append(replace(r, content=text))
            return out
        return results

    # -- strategy -----------------------------------------------------------------------------
    async def _top_documents(self, query: str, filters: RetrievalFilters, trace) -> frozenset[str]:
        """S-26: coarse step - rank documents by their summaries (kind=summary)."""
        with traced(trace, "document_search") as info:
            embedding = await self._retriever.embed(query)
            summaries = await self._store.search(
                embedding,
                filters,
                self._options.hierarchical_docs * 3,
                query_text=query if self._keyword else None,
                kinds=("summary",),
                hybrid=self._keyword,
            )
            doc_ids = list(dict.fromkeys(s.doc_id for s in summaries if s.doc_id))[
                : self._options.hierarchical_docs
            ]
            info["documents"] = doc_ids
        return frozenset(doc_ids)

    async def retrieve(
        self, query: str, filters: RetrievalFilters, top_k: int, history: History = ()
    ) -> RetrievalOutcome:
        outcome = RetrievalOutcome(contexts=[])
        options = self._options
        search_filters = filters
        if options.hierarchical and not filters.doc_ids:
            doc_ids = await self._top_documents(query, filters, outcome.trace)
            if doc_ids:
                search_filters = filters.with_changes(doc_ids=doc_ids)
        results = await self._search(query, search_filters, top_k, outcome.trace)

        if not results and options.fallback:
            with traced(outcome.trace, "fallback") as info:
                if filters.has_metadata_filters:
                    info["relaxed_filters"] = True
                    results = await self._search(query, filters.relaxed(), top_k, outcome.trace)
                if not results and self._transformer is not None:
                    broader = await self._transformer.step_back(query)
                    info["broadened_query"] = broader
                    if broader != query:
                        results = await self._search(
                            broader, filters.relaxed(), top_k, outcome.trace
                        )

        results = await self._map_multi_vector(results, filters)
        if options.recency_boost:
            results = sorted(
                (
                    replace(
                        r,
                        score=r.score
                        * recency_factor(
                            r.effective_date, options.recency_boost, options.recency_half_life_days
                        ),
                    )
                    for r in results
                ),
                key=lambda r: r.score,
                reverse=True,
            )
        if options.mmr:
            results = mmr(results, top_k, options.mmr_lambda)
        if options.expand != "none":
            with traced(outcome.trace, "expand", mode=options.expand) as info:
                results = await self._expand(results, filters)
                info["results"] = len(results)

        for context in results:
            context.metadata.setdefault("retriever", "vector")
        outcome.contexts = results[:top_k]
        return outcome

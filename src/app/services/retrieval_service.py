"""RAG orchestration shared by all strategies (A11 -> B|C|D -> A12 -> A13 -> A14).

input guardrail (C-156)
conversation history (C-157)          -> query understanding (C-158..C-161) + glossary (C-162)
shortcuts: chit-chat / out-of-scope (C-159, C-166)
answer cache (C-165)
strategy.retrieve with ACL + metadata filters (C-023, C-192)
drop passages carrying injected instructions (C-194)
context assembly: dedupe, rerank, cutoff, budget, ordering (A12)
generation from the versioned prompt (C-177..C-179), failover (C-190)
output guardrail (C-195), quality report (C-185, C-189)
persist turn (C-191), cache store, audit (C-199), metrics + rag_answer log (A16)
"""

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.core import metrics
from app.core.audit import audit
from app.core.auth import ANONYMOUS, UserContext
from app.core.config import RagMode
from app.core.exceptions import VectorBackendError
from app.core.logging import get_logger
from app.core.middleware import current_request_id
from app.core.usage import UsageTracker, track_usage
from app.models.schemas import (
    AnswerType,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    MatchedChunk,
    QualityReport,
    QueryUnderstanding,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    RetrievedContext,
    SearchFilters,
    TraceStepModel,
    Usage,
)
from app.rag.cache import AnswerCache
from app.rag.context.assembler import ContextAssembler
from app.rag.evaluation.quality import QualityAssessor
from app.rag.generation.llm_client import ChatClient
from app.rag.generation.prompts import (
    BLOCKED_INPUT_ANSWER,
    BLOCKED_OUTPUT_ANSWER,
    NO_CONTEXT_ANSWER,
    OUT_OF_SCOPE_ANSWER,
)
from app.rag.query.analysis import QueryAnalysis, QueryAnalyzer, heuristic_analysis
from app.rag.query.glossary import Glossary
from app.rag.safety import ContentSafety
from app.rag.strategies.base import RagStrategy, TraceStep, traced
from app.rag.strategies.registry import RagStrategyRegistry
from app.rag.types import RetrievalFilters
from app.rag.vectorstore.base import VectorSearchResult
from app.rag.vectorstore.multi import set_vector_backend
from app.services.conversations import ConversationStore, InMemoryConversationStore

logger = get_logger(__name__)

CHITCHAT_FALLBACK = "Hello! Ask me anything about the organisation's documents."


def _citations(contexts: list[VectorSearchResult], cited: list[int]) -> list[Citation]:
    # citations[n-1] corresponds to the [n] marker in the answer.
    return [
        Citation(
            index=i,
            chunk_id=c.chunk_id,
            source=c.source,
            score=c.score,
            title=c.title or c.metadata.get("title"),
            retriever=c.metadata.get("retriever"),
            cited=i in cited,
            doc_id=c.doc_id,
            section_path=c.section_path,
            page=c.page,
            effective_date=c.effective_date,
            url=c.metadata.get("url"),
            snippet=c.content[:240],
        )
        for i, c in enumerate(contexts, start=1)
    ]


def _retrieved(contexts: list[VectorSearchResult], cited: list[int]) -> list[RetrievedContext]:
    """Full passages behind the citations, for validation views (trace requests only)."""
    return [
        RetrievedContext(
            index=i,
            chunk_id=c.chunk_id,
            kind=c.kind,
            doc_id=c.doc_id,
            parent_id=c.parent_id,
            title=c.title or c.metadata.get("title", ""),
            section_path=c.section_path,
            source=c.source,
            page=c.page,
            chunk_index=c.chunk_index,
            effective_date=c.effective_date,
            score=c.score,
            retriever=c.metadata.get("retriever"),
            cited=i in cited,
            content=c.content,
            metadata=dict(c.metadata),
            matched_chunks=[MatchedChunk(**m) for m in c.matched_chunks],
        )
        for i, c in enumerate(contexts, start=1)
    ]


def _usage(tracker: UsageTracker) -> Usage:
    return Usage(
        prompt_tokens=tracker.prompt_tokens,
        completion_tokens=tracker.completion_tokens,
        total_tokens=tracker.total_tokens,
        llm_calls=tracker.llm_calls,
        by_model=dict(tracker.by_model),
        by_purpose=dict(tracker.by_purpose),
    )


def _trace_models(trace: list[TraceStep]) -> list[TraceStepModel]:
    return [TraceStepModel(step=s.step, detail=s.detail, duration_ms=s.duration_ms) for s in trace]


def build_filters(
    collection: str,
    user: UserContext,
    explicit: SearchFilters | None,
    analysis: QueryAnalysis | None,
) -> RetrievalFilters:
    """Caller ACL + explicit filters + filters read from the question (explicit wins)."""
    explicit = explicit or SearchFilters()
    analysed = analysis or QueryAnalysis("", "")
    return RetrievalFilters(
        collection=collection,
        principals=user.principals,
        doc_types=frozenset(explicit.doc_types or analysed.doc_types),
        languages=frozenset(explicit.languages or analysed.languages),
        sources=frozenset(explicit.sources),
        tags=frozenset(explicit.tags),
        effective_from=explicit.effective_from or analysed.effective_from,
        effective_to=explicit.effective_to or analysed.effective_to,
    )


def _understanding(analysis: QueryAnalysis, filters: RetrievalFilters) -> QueryUnderstanding:
    applied = {
        k: sorted(v) if isinstance(v, frozenset) else v
        for k, v in {
            "doc_types": filters.doc_types,
            "languages": filters.languages,
            "sources": filters.sources,
            "tags": filters.tags,
            "effective_from": filters.effective_from,
            "effective_to": filters.effective_to,
        }.items()
        if v
    }
    return QueryUnderstanding(
        standalone_question=analysis.standalone_question,
        intent=analysis.intent,
        complexity=analysis.complexity,
        applied_filters=applied,
        analysed_by=analysis.analysed_by,
    )


@dataclass
class Prepared:
    """Everything decided before retrieval, shared by the sync and streaming paths."""

    response_id: str
    started: float
    user: UserContext
    question: str  # what the user typed
    search_question: str  # standalone + glossary-expanded, used for retrieval
    analysis: QueryAnalysis
    filters: RetrievalFilters
    history: list[tuple[str, str]]
    conversation_id: str
    strategy: RagStrategy | None = None
    vector_store: str = ""  # backend that served this request's retrieval
    shortcut: AnswerType | None = None
    shortcut_answer: str = ""
    trace: list[TraceStep] = field(default_factory=list)
    usage: UsageTracker = field(default_factory=UsageTracker)
    cache_scope: str = ""
    cache_embedding: list[float] | None = None
    outcome: object | None = None  # RetrievalOutcome of the strategy (agent answer, flags)

    @property
    def mode(self) -> RagMode:
        return self.strategy.mode if self.strategy else RagMode.SEMANTIC


@dataclass
class StreamResult:
    prepared: Prepared
    contexts: list[VectorSearchResult]
    tokens: AsyncIterator[str]
    cached: dict | None = None

    @property
    def rag_mode(self) -> RagMode:
        return self.prepared.mode

    @property
    def response_id(self) -> str:
        return self.prepared.response_id

    @property
    def usage(self) -> UsageTracker:
        return self.prepared.usage

    @property
    def trace(self) -> list[TraceStepModel]:
        return _trace_models(self.prepared.trace)

    @property
    def citations(self) -> list[Citation]:
        return _citations(self.contexts, [])


@dataclass
class StreamSummary:
    quality: QualityReport | None
    usage: Usage
    citations: list[Citation]
    latency_ms: float
    answer_type: AnswerType
    withheld: bool = False
    replacement: str = ""


class RetrievalService:
    def __init__(
        self,
        registry: RagStrategyRegistry,
        chat_client: ChatClient,
        assessor: QualityAssessor | None = None,
        *,
        assembler: ContextAssembler | None = None,
        analyzer: QueryAnalyzer | None = None,
        safety: ContentSafety | None = None,
        glossary: Glossary | None = None,
        cache: AnswerCache | None = None,
        conversations: ConversationStore | None = None,
        embedder=None,
        max_history_turns: int = 6,
        log_question_text: bool = False,
        escalation: bool = False,
        max_escalations: int = 1,
        vector_backends: Sequence[str] = (),
        default_vector_backend: str = "",
    ) -> None:
        self._registry = registry
        self._chat_client = chat_client
        self._assessor = assessor or QualityAssessor(judge=None)
        self._assembler = assembler or ContextAssembler()
        self._analyzer = analyzer
        self._safety = safety or ContentSafety(mode="off")
        self._glossary = glossary or Glossary()
        self._cache = cache
        self._conversations = conversations or InMemoryConversationStore()
        self._embedder = embedder
        self._max_turns = max_history_turns
        self._log_question_text = log_question_text
        self._escalation = escalation
        self._max_escalations = max_escalations
        self._vector_backends = tuple(vector_backends)
        self._default_vector_backend = default_vector_backend

    @property
    def registry(self) -> RagStrategyRegistry:
        return self._registry

    @property
    def vector_backends(self) -> tuple[str, ...]:
        return self._vector_backends

    @property
    def default_vector_backend(self) -> str:
        return self._default_vector_backend

    def _select_vector_backend(self, requested: str | None, force: bool) -> str:
        """Validate the request's vector store and make retrieval use it (this request only)."""
        if not self._vector_backends:
            if requested:
                raise VectorBackendError("No vector store is enabled on this deployment")
            set_vector_backend(None)
            return ""
        name = requested or self._default_vector_backend or self._vector_backends[0]
        if name not in self._vector_backends:
            enabled = ", ".join(self._vector_backends)
            raise VectorBackendError(f"Vector store '{name}' is not enabled (enabled: {enabled})")
        if name != self._default_vector_backend and not (self._registry.allow_override or force):
            raise VectorBackendError(
                "Per-request vector_store is disabled (RAG_ALLOW_MODE_OVERRIDE)"
            )
        set_vector_backend(name)
        return name

    def _prompt_version(self) -> str:
        return (
            self._chat_client.prompt("rag_answer").version
            if hasattr(self._chat_client, "prompt")
            else ""
        )

    # -- preparation (A11) ----------------------------------------------------------------------
    async def _prepare(
        self,
        *,
        query: str,
        collection: str,
        user: UserContext,
        rag_mode: RagMode | None,
        filters: SearchFilters | None,
        history: list[ChatMessage] | None = None,
        conversation_id: str | None = None,
        for_chat: bool = True,
        force_mode: bool = False,
        vector_store: str | None = None,
    ) -> Prepared:
        trace: list[TraceStep] = []
        conversation_id = conversation_id or uuid.uuid4().hex
        backend = self._select_vector_backend(vector_store, force_mode)
        prepared = Prepared(
            response_id=current_request_id(),
            started=time.perf_counter(),
            user=user,
            question=query,
            search_question=query,
            analysis=heuristic_analysis(query),
            filters=build_filters(collection, user, filters, None),
            history=[],
            conversation_id=conversation_id,
            trace=trace,
            vector_store=backend,
        )
        with track_usage(prepared.usage):
            if self._safety.enabled:
                with traced(trace, "input_guardrail") as info:
                    verdict = await self._safety.check_input(query)
                    info["blocked"] = verdict.blocked
                if verdict.blocked:
                    logger.warning("input_blocked", reasons=verdict.reasons)
                    prepared.shortcut, prepared.shortcut_answer = "blocked", BLOCKED_INPUT_ANSWER
                    return prepared

            prepared.history = [(m.role, m.content) for m in history or []]
            if not prepared.history and conversation_id and for_chat:
                prepared.history = await self._conversations.history(
                    conversation_id, user.user_id, self._max_turns
                )

            if self._analyzer is not None:
                with traced(trace, "query_analysis") as info:
                    prepared.analysis = await self._analyzer.analyze(query, prepared.history)
                    info.update(
                        intent=prepared.analysis.intent,
                        complexity=prepared.analysis.complexity,
                        standalone=prepared.analysis.standalone_question,
                    )
            analysis = prepared.analysis
            prepared.filters = build_filters(collection, user, filters, analysis)
            prepared.search_question = self._glossary.expand(analysis.standalone_question)
            prepared.strategy = self._registry.resolve(
                rag_mode, analysis.complexity, force=force_mode
            )
            prepared.trace.append(  # I-01: which approach answered and why
                TraceStep(
                    "router",
                    {
                        "requested": str(rag_mode or self._registry.default_mode),
                        "complexity": analysis.complexity,
                        "mode": str(prepared.mode),
                        **({"vector_store": backend} if backend else {}),
                    },
                )
            )

            if for_chat and analysis.intent == "chitchat":
                prepared.shortcut = "chitchat"
                try:
                    prepared.shortcut_answer = await self._chat_client.complete_text(
                        "chitchat", purpose="chitchat", question=query
                    )
                except Exception:  # noqa: BLE001 - a canned reply is fine here
                    prepared.shortcut_answer = CHITCHAT_FALLBACK
            elif for_chat and analysis.intent == "out_of_scope":
                prepared.shortcut, prepared.shortcut_answer = "out_of_scope", OUT_OF_SCOPE_ANSWER
        return prepared

    async def _contexts(self, prepared: Prepared, top_k: int) -> list[VectorSearchResult]:
        """Strategy retrieval, injection screening (C-194) and shared context assembly."""
        strategy = prepared.strategy
        outcome = await strategy.retrieve(
            prepared.search_question, prepared.filters, top_k, prepared.history
        )
        prepared.trace.extend(outcome.trace)
        prepared.outcome = outcome
        contexts = outcome.contexts
        metrics.retrieval_contexts.record(len(contexts), {"rag.mode": str(strategy.mode)})
        if self._safety.enabled and contexts:
            with traced(prepared.trace, "context_guardrail") as info:
                flags = await self._safety.flag_documents([c.content for c in contexts])
                dropped = [c.chunk_id for c, bad in zip(contexts, flags, strict=True) if bad]
                contexts = [c for c, bad in zip(contexts, flags, strict=True) if not bad]
                info["dropped"] = dropped
                if dropped:
                    logger.warning("context_injection_dropped", chunks=dropped)
                    outcome.answer = None  # a pre-generated answer may rely on a dropped passage
        if outcome.answer is not None:
            # The agent already answered from these passages in this order: keep [n] stable.
            return contexts
        return await self._assembler.assemble(prepared.search_question, contexts, prepared.trace)

    async def _cache_lookup(self, prepared: Prepared) -> dict | None:
        if self._cache is None or prepared.shortcut:
            return None
        prepared.cache_scope = AnswerCache.scope(
            prepared.filters.signature(),
            str(prepared.mode),
            self._prompt_version(),
            prepared.vector_store,
        )
        if self._embedder is not None:
            prepared.cache_embedding = await self._embedder.embed(prepared.search_question)
        with traced(prepared.trace, "cache_lookup") as info:
            hit = await self._cache.lookup(
                prepared.cache_scope, prepared.search_question, prepared.cache_embedding
            )
            info["hit"] = hit is not None
        return hit

    # -- public API ----------------------------------------------------------------------------
    async def retrieve(
        self, request: RetrievalRequest, user: UserContext = ANONYMOUS, top_k: int = 8
    ) -> RetrievalResponse:
        prepared = await self._prepare(
            query=request.query,
            collection=request.collection,
            user=user,
            rag_mode=request.rag_mode,
            filters=request.filters,
            for_chat=False,
            vector_store=request.vector_store,
        )
        contexts: list[VectorSearchResult] = []
        if prepared.shortcut != "blocked":
            with track_usage(prepared.usage):
                contexts = await self._contexts(prepared, top_k)
        self._record_latency(prepared.started, prepared.mode, "retrieve", prepared.vector_store)
        audit(
            "retrieve",
            user_id=user.user_id,
            response_id=prepared.response_id,
            rag_mode=str(prepared.mode),
            vector_store=prepared.vector_store,
            collection=request.collection,
            chunks=[c.chunk_id for c in contexts],
            outcome=prepared.shortcut or "ok",
        )
        return RetrievalResponse(
            query=request.query,
            rag_mode=prepared.mode,
            vector_store=prepared.vector_store,
            results=[
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    source=r.source,
                    metadata=r.metadata,
                    doc_id=r.doc_id,
                    title=r.title,
                    section_path=r.section_path,
                    page=r.page,
                    effective_date=r.effective_date,
                )
                for r in contexts
            ],
            understanding=_understanding(prepared.analysis, prepared.filters),
            trace=_trace_models(prepared.trace) if request.include_trace else None,
        )

    async def chat(
        self,
        request: ChatRequest,
        user: UserContext = ANONYMOUS,
        top_k: int = 8,
        *,
        use_cache: bool = True,
        force_mode: bool = False,
    ) -> ChatResponse:
        """`use_cache=False` neither reads nor writes the answer cache and `force_mode` honours
        `rag_mode` even when overrides are disabled - both for the admin playground, which
        compares modes and models on fresh answers."""
        prepared = await self._prepare(
            query=request.query,
            collection=request.collection,
            user=user,
            rag_mode=request.rag_mode,
            filters=request.filters,
            history=request.history,
            conversation_id=request.conversation_id,
            force_mode=force_mode,
            vector_store=request.vector_store,
        )
        contexts: list[VectorSearchResult] = []
        quality: QualityReport | None = None
        answer_type: AnswerType = prepared.shortcut or "answer"
        answer = prepared.shortcut_answer

        with track_usage(prepared.usage):
            if not prepared.shortcut:
                cached = await self._cache_lookup(prepared) if use_cache else None
                if cached is not None:
                    return self._from_cache(prepared, cached, request)
                judged = self._assessor.should_judge(request.evaluate)
                answer, answer_type, contexts, quality = await self._answer(prepared, top_k, judged)
                escalations = 0
                while (  # I-03: escalate low-confidence answers to a more capable mode
                    self._escalation
                    and escalations < self._max_escalations
                    and answer_type in ("answer", "not_found")
                    and quality is not None
                    and quality.confidence == "low"
                    and (stronger := self._registry.escalate(prepared.mode)) is not None
                ):
                    escalations += 1
                    prepared.trace.append(
                        TraceStep(
                            "escalation", {"from": str(prepared.mode), "to": str(stronger.mode)}
                        )
                    )
                    prepared.strategy = stronger
                    answer, answer_type, contexts, quality = await self._answer(
                        prepared, top_k, judged
                    )
                if not judged and answer_type == "answer":
                    self._assessor.sample_in_background(
                        prepared.analysis.standalone_question,
                        answer,
                        contexts,
                        rag_mode=prepared.mode,
                        response_id=prepared.response_id,
                    )

        latency_ms = self._record_latency(
            prepared.started, prepared.mode, "chat", prepared.vector_store
        )
        response = ChatResponse(
            response_id=prepared.response_id,
            answer=answer,
            answer_type=answer_type,
            citations=_citations(contexts, quality.signals.cited_passages if quality else []),
            conversation_id=prepared.conversation_id,
            rag_mode=prepared.mode,
            vector_store=prepared.vector_store,
            review_required=self._review_required(prepared, quality),
            prompt_version=self._prompt_version(),
            understanding=_understanding(prepared.analysis, prepared.filters),
            quality=quality,
            usage=_usage(prepared.usage),
            latency_ms=latency_ms,
            trace=_trace_models(prepared.trace) if request.include_trace else None,
            retrieved=(
                _retrieved(contexts, quality.signals.cited_passages if quality else [])
                if request.include_trace
                else None
            ),
        )
        await self._finish(prepared, response, contexts)
        return response

    async def _answer(
        self, prepared: Prepared, top_k: int, judged: bool
    ) -> tuple[str, AnswerType, list[VectorSearchResult], QualityReport | None]:
        """Retrieve with the current strategy, answer, guard and assess."""
        answer_type: AnswerType = "answer"
        contexts = await self._contexts(prepared, top_k)
        outcome = prepared.outcome
        if outcome is not None and outcome.clarification:  # A-45
            return outcome.clarification, "clarification", [], None
        if outcome is not None and outcome.answer and contexts:  # agent answered (A-28)
            answer = outcome.answer
        elif contexts:
            answer = await self._chat_client.generate_answer(
                prepared.analysis.standalone_question, contexts, prepared.history
            )
        else:
            answer, answer_type = NO_CONTEXT_ANSWER, "not_found"
        answer, answer_type, contexts = await self._guard_output(
            prepared, answer, answer_type, contexts
        )
        quality = await self._assessor.assess(
            prepared.analysis.standalone_question,
            answer,
            contexts,
            rag_mode=prepared.mode,
            run_judge=judged and answer_type == "answer",
        )
        return answer, answer_type, contexts, quality

    async def chat_stream(
        self, request: ChatRequest, user: UserContext = ANONYMOUS, top_k: int = 8
    ) -> StreamResult:
        """Everything up to generation happens here (errors become normal HTTP errors);
        the endpoint streams `tokens` and then calls finish_stream."""
        prepared = await self._prepare(
            query=request.query,
            collection=request.collection,
            user=user,
            rag_mode=request.rag_mode,
            filters=request.filters,
            history=request.history,
            conversation_id=request.conversation_id,
            vector_store=request.vector_store,
        )

        async def fixed(text: str) -> AsyncIterator[str]:
            yield text

        if prepared.shortcut:
            return StreamResult(prepared, [], fixed(prepared.shortcut_answer))
        with track_usage(prepared.usage):
            cached = await self._cache_lookup(prepared)
            if cached is not None:
                return StreamResult(prepared, [], fixed(cached["answer"]), cached=cached)
            contexts = await self._contexts(prepared, top_k)
        outcome = prepared.outcome
        if outcome is not None and outcome.clarification:
            prepared.shortcut = "clarification"
            return StreamResult(prepared, [], fixed(outcome.clarification))
        if not contexts:
            prepared.shortcut = "not_found"
            return StreamResult(prepared, [], fixed(NO_CONTEXT_ANSWER))
        if outcome is not None and outcome.answer:
            return StreamResult(prepared, contexts, fixed(outcome.answer))
        tokens = self._chat_client.stream_answer(
            prepared.analysis.standalone_question, contexts, prepared.history
        )
        return StreamResult(prepared, contexts, tokens)

    async def finish_stream(
        self, result: StreamResult, answer: str, evaluate: bool = False
    ) -> StreamSummary:
        prepared = result.prepared
        answer_type: AnswerType = prepared.shortcut or "answer"
        quality = None
        contexts = result.contexts
        withheld, replacement = False, ""
        with track_usage(prepared.usage):
            if answer_type == "answer" and result.cached is None:
                guarded, answer_type, contexts = await self._guard_output(
                    prepared, answer, answer_type, contexts
                )
                # Tokens were already sent: tell the client to replace them (C-195).
                withheld = guarded != answer
                replacement = guarded if withheld else ""
                answer = guarded
            if answer_type in ("answer", "not_found") and result.cached is None:
                judged = self._assessor.should_judge(evaluate)
                quality = await self._assessor.assess(
                    prepared.analysis.standalone_question,
                    answer,
                    contexts,
                    rag_mode=prepared.mode,
                    run_judge=judged and answer_type == "answer",
                )
        if result.cached is not None:
            quality = (
                QualityReport.model_validate(result.cached["quality"])
                if result.cached.get("quality")
                else None
            )
        latency_ms = self._record_latency(
            prepared.started, prepared.mode, "chat_stream", prepared.vector_store
        )
        citations = _citations(contexts, quality.signals.cited_passages if quality else [])
        response = ChatResponse(
            response_id=prepared.response_id,
            answer=answer,
            answer_type=answer_type,
            citations=citations,
            conversation_id=prepared.conversation_id,
            rag_mode=prepared.mode,
            vector_store=prepared.vector_store,
            cached=result.cached is not None,
            prompt_version=self._prompt_version(),
            quality=quality,
            usage=_usage(prepared.usage),
            latency_ms=latency_ms,
        )
        if result.cached is None:
            await self._finish(prepared, response, contexts)
        return StreamSummary(
            quality=quality,
            usage=_usage(prepared.usage),
            citations=citations,
            latency_ms=latency_ms,
            answer_type=answer_type,
            withheld=withheld,
            replacement=replacement,
        )

    @staticmethod
    def _review_required(prepared: Prepared, quality: QualityReport | None) -> bool:
        """A-44: flag answers a person should check (failed agent verification)."""
        outcome = prepared.outcome
        return bool(outcome is not None and getattr(outcome, "review_required", False))

    # -- steps after generation (A13, A14) -------------------------------------------------------
    async def _guard_output(
        self,
        prepared: Prepared,
        answer: str,
        answer_type: AnswerType,
        contexts: list[VectorSearchResult],
    ) -> tuple[str, AnswerType, list[VectorSearchResult]]:
        if not self._safety.enabled or answer_type != "answer":
            return answer, answer_type, contexts
        with traced(prepared.trace, "output_guardrail") as info:
            # PII that already appears in the question or the sources is not a leak.
            allowed = prepared.question + "\n" + "\n".join(c.content for c in contexts)
            verdict = await self._safety.check_output(answer, allowed_pii=allowed)
            info["blocked"] = verdict.blocked
        if verdict.blocked:
            logger.warning("output_blocked", reasons=verdict.reasons)
            return BLOCKED_OUTPUT_ANSWER, "blocked", []
        return answer, answer_type, contexts

    def _from_cache(self, prepared: Prepared, cached: dict, request: ChatRequest) -> ChatResponse:
        latency_ms = self._record_latency(
            prepared.started, prepared.mode, "chat_cached", prepared.vector_store
        )
        response = ChatResponse.model_validate(
            {
                **cached,
                "response_id": prepared.response_id,
                "conversation_id": prepared.conversation_id,
                "vector_store": prepared.vector_store,
                "cached": True,
                "latency_ms": latency_ms,
                "usage": _usage(prepared.usage).model_dump(),
                "trace": _trace_models(prepared.trace) if request.include_trace else None,
            }
        )
        audit(
            "chat",
            user_id=prepared.user.user_id,
            response_id=prepared.response_id,
            rag_mode=str(prepared.mode),
            cached=True,
        )
        return response

    async def _finish(
        self, prepared: Prepared, response: ChatResponse, contexts: list[VectorSearchResult]
    ) -> None:
        sources = sorted({c.source for c in contexts})
        await self._conversations.append(
            prepared.conversation_id,
            prepared.user.user_id,
            prepared.question,
            response.answer,
            response_id=prepared.response_id,
            rag_mode=str(prepared.mode),
            sources=sources,
        )
        if (
            self._cache is not None
            and prepared.cache_scope
            and response.answer_type == "answer"
            and response.quality is not None
            and response.quality.confidence != "low"
        ):
            payload = response.model_dump(
                include={
                    "answer",
                    "answer_type",
                    "citations",
                    "rag_mode",
                    "prompt_version",
                    "quality",
                    "understanding",
                },
                mode="json",
            )
            await self._cache.store(
                prepared.cache_scope, prepared.search_question, prepared.cache_embedding, payload
            )
        audit(
            "chat",
            user_id=prepared.user.user_id,
            response_id=prepared.response_id,
            conversation_id=prepared.conversation_id,
            rag_mode=str(prepared.mode),
            vector_store=prepared.vector_store,
            collection=prepared.filters.collection,
            answer_type=response.answer_type,
            chunks=[c.chunk_id for c in contexts],
            confidence=response.quality.confidence if response.quality else None,
            review_required=response.review_required,
            **({"question": prepared.question} if self._log_question_text else {}),
        )
        self._log_answer(prepared, response)

    # -- telemetry ----------------------------------------------------------------------------
    @staticmethod
    def _record_latency(
        started: float, mode: RagMode, operation: str, vector_store: str = ""
    ) -> float:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        attributes = {"rag.mode": str(mode), "rag.operation": operation}
        if vector_store:
            attributes["rag.vector_store"] = vector_store
        metrics.request_duration.record(latency_ms, attributes)
        return latency_ms

    def _log_answer(self, prepared: Prepared, response: ChatResponse) -> None:
        """One queryable line per answer (App Insights: traces | where message == "rag_answer")."""
        quality = response.quality
        judge = quality.judge if quality else None
        usage = prepared.usage
        logger.info(
            "rag_answer",
            response_id=prepared.response_id,
            rag_mode=str(prepared.mode),
            vector_store=prepared.vector_store,
            collection=prepared.filters.collection,
            answer_type=response.answer_type,
            intent=prepared.analysis.intent,
            complexity=prepared.analysis.complexity,
            latency_ms=response.latency_ms,
            confidence=quality.confidence if quality else None,
            answered=quality.signals.answered if quality else None,
            contexts=quality.signals.contexts_retrieved if quality else 0,
            cited=len(quality.signals.cited_passages) if quality else 0,
            invalid_citations=len(quality.signals.invalid_citations) if quality else 0,
            citation_coverage=quality.signals.citation_coverage if quality else None,
            groundedness=judge.groundedness if judge else None,
            answer_relevance=judge.answer_relevance if judge else None,
            context_relevance=judge.context_relevance if judge else None,
            prompt_version=response.prompt_version,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            llm_calls=usage.llm_calls,
            tokens_by_purpose=usage.by_purpose,
            **({"question": prepared.question} if self._log_question_text else {}),
        )

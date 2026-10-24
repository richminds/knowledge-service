from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import re

from pydantic import BaseModel, Field, field_validator

from app.core.config import IngestionTarget, RagMode

# Collections become part of search filters and chunk ids; keep them to a safe charset.
CollectionName = Field(default="default", pattern=r"^[A-Za-z0-9_\-]{1,64}$")


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionRequest(BaseModel):
    source_uri: str = Field(
        ...,
        description="Path under INGESTION_LOCAL_ROOT, or blob://<container>/<prefix>",
        examples=["policies/", "blob://documents/hr/"],
    )
    collection: str = CollectionName
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Stored on every chunk (e.g. department, owner)"
    )
    targets: list[IngestionTarget] | None = Field(
        default=None,
        description="Stores to write: vector and/or graph. Defaults to INGESTION_TARGETS.",
    )
    acl: list[str] | None = Field(
        default=None,
        max_length=200,
        description="Principal ids (Entra user/group object ids, or * for everyone) allowed "
        "to read these documents. Defaults to the source's own permissions, else "
        "INGESTION_DEFAULT_ACL.",
    )
    full_sync: bool = Field(
        default=False,
        description="Also delete documents previously ingested from this source URI that "
        "no longer exist there (C-144).",
    )
    force: bool = Field(default=False, description="Re-process documents even if unchanged.")


class IngestionResponse(BaseModel):
    job_id: str
    status: JobStatus


class IngestionJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: str | None = None
    targets: list[str] = Field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    unchanged: int = 0
    acl_updated: int = 0
    deleted: int = 0
    duplicates: dict[str, str] = Field(default_factory=dict)
    quarantined: dict[str, str] = Field(default_factory=dict)
    failed_sources: dict[str, str] = Field(default_factory=dict)
    reconciliation: dict[str, int] = Field(default_factory=dict)
    created_at: datetime | None = None
    finished_at: datetime | None = None


class ManifestEntryModel(BaseModel):
    source: str
    doc_id: str
    status: str
    title: str = ""
    chunk_count: int = 0
    entity_count: int = 0
    acl: list[str] = Field(default_factory=list)
    duplicate_of: str = ""
    error: str = ""
    reasons: list[str] = Field(default_factory=list)
    run_id: str = ""
    updated_at: str = ""


class UploadIngestionResponse(BaseModel):
    source: str
    collection: str
    targets: list[str]
    chunk_count: int
    entity_count: int = 0
    relation_count: int = 0


class DeleteCollectionResponse(BaseModel):
    collection: str
    deleted_chunks: int
    deleted_graph_nodes: int = 0


RagModeField = Field(
    default=None,
    description="semantic (vector) | hybrid (vector + keyword) | hybrid_graph (vector + keyword "
    "+ knowledge graph) | graph | agentic | auto. Defaults to RAG_MODE.",
)
IncludeTraceField = Field(
    default=False, description="Return the retrieval trace (steps, timings) for validation."
)
VectorStoreField = Field(
    default=None,
    description="Vector store that serves retrieval: mongo. "
    "Defaults to VECTOR_STORE_PROVIDER; see GET /rag/modes for what is enabled.",
    max_length=32,
)


class TraceStepModel(BaseModel):
    step: str
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class RagModesResponse(BaseModel):
    default: RagMode
    enabled: list[RagMode]
    allow_override: bool
    vector_stores: list[str] = Field(default_factory=list, description="Selectable vector stores")
    default_vector_store: str = ""


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_query(value: str) -> str:
    """C-155: drop control characters, collapse whitespace, reject blank questions."""
    cleaned = re.sub(r"[ \t]+", " ", _CONTROL_CHARS.sub("", value)).strip()
    if not cleaned:
        raise ValueError("query must contain text")
    return cleaned


class SearchFilters(BaseModel):
    """Explicit metadata filters from the caller (merged with ones read from the question)."""

    doc_types: list[str] = Field(default_factory=list, max_length=10)
    languages: list[str] = Field(default_factory=list, max_length=5)
    sources: list[str] = Field(default_factory=list, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=10)
    effective_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    effective_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class QueryUnderstanding(BaseModel):
    """How the question was interpreted (for validation / debugging in the UI)."""

    standalone_question: str
    intent: Literal["rag", "chitchat", "out_of_scope"]
    complexity: Literal["simple", "relationship", "global", "multi_step"]
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    analysed_by: Literal["llm", "heuristic"] = "heuristic"


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    collection: str = CollectionName
    top_k: int | None = Field(default=None, ge=1, le=50)
    rag_mode: RagMode | None = RagModeField
    vector_store: str | None = VectorStoreField
    include_trace: bool = IncludeTraceField
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        return clean_query(value)


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)
    doc_id: str = ""
    title: str = ""
    section_path: str = ""
    page: int | None = None
    effective_date: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    rag_mode: RagMode
    vector_store: str = ""
    results: list[RetrievedChunk]
    understanding: QueryUnderstanding | None = None
    trace: list[TraceStepModel] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    collection: str = CollectionName
    top_k: int | None = Field(default=None, ge=1, le=50)
    conversation_id: str | None = None
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=20, description="Prior turns, oldest first"
    )
    rag_mode: RagMode | None = RagModeField
    vector_store: str | None = VectorStoreField
    include_trace: bool = IncludeTraceField
    evaluate: bool = Field(
        default=False,
        description="Also score the answer with the LLM judge (adds one LLM call, ~1-3 s). "
        "Honoured when QUALITY_JUDGE=request; always on when QUALITY_JUDGE=always.",
    )
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        return clean_query(value)


class Citation(BaseModel):
    index: int = Field(..., description="The [n] marker used in the answer")
    chunk_id: str
    source: str
    score: float
    title: str | None = None
    retriever: str | None = Field(default=None, description="vector or graph")
    cited: bool = Field(default=False, description="Whether the answer actually cites [n]")
    doc_id: str = ""
    section_path: str = ""
    page: int | None = None
    effective_date: str | None = None
    url: str | None = Field(default=None, description="Link to the source, when known")
    snippet: str = Field(default="", description="Start of the passage, for source previews")


class MatchedChunk(BaseModel):
    """A chunk the search matched; with parent expansion, the parent is what gets used."""

    chunk_id: str
    score: float = 0.0
    preview: str = ""


class RetrievedContext(BaseModel):
    """A passage the answer was grounded on, in full, with its index metadata (trace only)."""

    index: int = Field(..., description="The [n] marker; same order as citations")
    chunk_id: str
    kind: str = Field(default="chunk", description="chunk, or parent when expanded to a section")
    doc_id: str = ""
    parent_id: str = ""
    title: str = ""
    section_path: str = ""
    source: str = ""
    page: int | None = None
    chunk_index: int = 0
    effective_date: str | None = None
    score: float = 0.0
    retriever: str | None = None
    cited: bool = False
    content: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    matched_chunks: list[MatchedChunk] = Field(
        default_factory=list,
        description="For a parent section: the child chunks the search matched inside it",
    )


# -- Answer quality ------------------------------------------------------------------------
Confidence = Literal["high", "medium", "low"]


class QualitySignals(BaseModel):
    """Computed from the answer text on every response, at no extra LLM cost."""

    answered: bool = Field(..., description="False if the assistant said it could not answer")
    contexts_retrieved: int
    cited_passages: list[int] = Field(description="Passage numbers the answer cites")
    invalid_citations: list[int] = Field(
        description="Cited numbers that don't exist (a sign of hallucinated references)"
    )
    citation_coverage: float = Field(
        ..., ge=0, le=1, description="Share of answer statements that cite a source"
    )
    cited_context_ratio: float = Field(
        ..., ge=0, le=1, description="Share of retrieved passages the answer used"
    )


class JudgeScores(BaseModel):
    """LLM-as-judge scores (1 = poor, 5 = excellent) - the "RAG triad"."""

    groundedness: int = Field(..., ge=1, le=5, description="Claims supported by the sources")
    answer_relevance: int = Field(..., ge=1, le=5, description="Answers the question asked")
    context_relevance: int = Field(
        ..., ge=1, le=5, description="Retrieved sources fit the question"
    )
    reasoning: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    confidence: Confidence = Field(..., description="Overall, for display to the user")
    explanation: str = Field(..., description="Plain-language reason for the confidence")
    evaluated_by: Literal["signals", "signals+llm_judge"]
    signals: QualitySignals
    judge: JudgeScores | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    by_model: dict[str, int] = Field(default_factory=dict, description="Tokens per deployment")
    by_purpose: dict[str, int] = Field(
        default_factory=dict, description="Tokens per call type (answer, query_analysis, ...)"
    )


AnswerType = Literal["answer", "not_found", "chitchat", "out_of_scope", "blocked", "clarification"]


class ChatResponse(BaseModel):
    response_id: str = Field(..., description="Use with POST /feedback; equals X-Request-ID")
    answer: str
    answer_type: AnswerType = "answer"
    citations: list[Citation]
    conversation_id: str | None = None
    rag_mode: RagMode
    vector_store: str = Field(default="", description="Vector store that served retrieval")
    cached: bool = False
    review_required: bool = Field(
        default=False, description="The agent could not verify the answer; have a person review it"
    )
    prompt_version: str = ""
    understanding: QueryUnderstanding | None = None
    quality: QualityReport | None = None
    usage: Usage | None = None
    latency_ms: float = 0.0
    trace: list[TraceStepModel] | None = None
    retrieved: list[RetrievedContext] | None = Field(
        default=None,
        description="Full retrieved passages with metadata; returned with include_trace only",
    )


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)
    rag_mode: RagMode | None = None


class FeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"


# -- Admin RAG studio ----------------------------------------------------------------------
class AdminGoldenCase(BaseModel):
    id: str
    question: str
    collection: str = "default"
    question_type: str = "factoid"
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer: str = ""


class AdminRagConfig(BaseModel):
    """What the admin playground can choose from, and the targets it scores against."""

    modes: RagModesResponse
    answer_models: list[str] = Field(description="Deployments selectable for answers")
    default_answer_model: str
    models: dict[str, str] = Field(
        description="Every model in play: utility, embedding, reranker, judge, vision, ..."
    )
    retrieval: dict[str, Any]
    quality_judge: str
    ingestion: dict[str, Any]
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    golden_set: list[AdminGoldenCase] = Field(default_factory=list)


class AdminEvalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    collection: str = CollectionName
    rag_mode: RagMode | None = RagModeField
    vector_store: str | None = VectorStoreField
    model: str | None = Field(default=None, description="Answer deployment; see /admin/rag/config")
    top_k: int | None = Field(default=None, ge=1, le=50)
    judge: bool = Field(default=True, description="Score with the LLM judge (RAG triad)")
    filters: SearchFilters | None = None
    expected_sources: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Source suffixes that should be retrieved -> hit/recall/MRR/nDCG",
    )
    expected_answer: str = Field(
        default="", max_length=8000, description="Reference answer -> correctness (1-5)"
    )

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        return clean_query(value)


class AdminEvalResponse(BaseModel):
    model: str = Field(description="Answer deployment used")
    top_k: int
    response: ChatResponse
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Retrieval (hit, recall, precision, mrr, ndcg), citation precision/recall, "
        "judge scores, correctness, latency_ms, tokens, agent trajectory",
    )
    threshold_violations: list[str] = Field(default_factory=list)


class VectorStoreStatus(BaseModel):
    name: str
    default: bool = False
    count: int | None = Field(default=None, description="Records in the collection (None = error)")
    latency_ms: float = Field(default=0.0, description="Time the count query took")
    error: str = ""


class VectorStoreStatusResponse(BaseModel):
    """Per-store record counts, so a quality comparison between stores is like for like."""

    collection: str
    stores: list[VectorStoreStatus]
    in_sync: bool = Field(description="Every store answered and holds the same record count")

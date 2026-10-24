"""Custom OpenTelemetry metrics and tracer for the RAG pipeline.

Exported to Application Insights (customMetrics table) when telemetry is configured; the
OpenTelemetry API makes every call a no-op otherwise. Dimensions are kept low-cardinality
(mode, step, metric name), never user text or ids.

| Metric                   | Type      | Dimensions                         |
|--------------------------|-----------|------------------------------------|
| rag.request.duration     | histogram | rag.mode, rag.operation            |
| rag.step.duration        | histogram | rag.step                           |
| rag.retrieval.contexts   | histogram | rag.mode                           |
| rag.quality.score        | histogram | rag.mode, quality.metric, evaluator|
| rag.answers              | counter   | rag.mode, confidence, answered     |
| rag.llm.tokens           | counter   | llm.purpose, llm.model, token.type |
| rag.feedback             | counter   | rag.mode, rating                   |
"""

from opentelemetry import metrics, trace

tracer = trace.get_tracer("app.rag")
_meter = metrics.get_meter("app.rag")

request_duration = _meter.create_histogram(
    "rag.request.duration", unit="ms", description="End-to-end retrieval/chat latency"
)
step_duration = _meter.create_histogram(
    "rag.step.duration", unit="ms", description="Latency of one RAG step (search, grade, ...)"
)
retrieval_contexts = _meter.create_histogram(
    "rag.retrieval.contexts", description="Contexts passed to answer generation"
)
quality_score = _meter.create_histogram(
    "rag.quality.score", description="Answer quality scores (signals 0-1, judge 1-5)"
)
answers = _meter.create_counter("rag.answers", description="Answers by confidence level")
llm_tokens = _meter.create_counter("rag.llm.tokens", unit="{token}", description="LLM tokens")
feedback = _meter.create_counter("rag.feedback", description="User feedback on answers")

# Ingestion (C-149)
ingest_documents = _meter.create_counter(
    "rag.ingest.documents", description="Documents processed by outcome (indexed/skipped/...)"
)
ingest_chunks = _meter.create_counter("rag.ingest.chunks", description="Chunks written")
ingest_duration = _meter.create_histogram(
    "rag.ingest.document_duration", unit="ms", description="Per-document ingestion latency"
)

import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Tests must never reach real Azure services or MongoDB, whatever is in a developer's .env or shell.
_PREFIXES = (
    "AZURE_", "GRAPH_", "RAG_", "INGESTION_", "API_KEY", "APPLICATIONINSIGHTS", "AUTH_",
    "ENTRA_", "CONTENT_SAFETY", "RERANKER", "COSMOS_", "REDIS_", "ANSWER_CACHE", "QUALITY_",
    "SEMANTIC_", "CONVERSATION_", "PROMPT_", "VECTOR_", "MONGODB_", "MANIFEST_",
)  # fmt: skip
for var in list(os.environ):
    if var.startswith(_PREFIXES):
        del os.environ[var]
os.environ.update(
    {
        "APP_ENV": "test",
        "APP_ENV_FILE": "__tests_do_not_read_dotenv__",
        # Dummy endpoints so config validation passes; every client is a fake below.
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "MONGODB_URL": "mongodb://test.invalid:27017",
        "RAG_ENABLED_MODES": "semantic",
        # The upload tests use small Markdown files; the default list has no .md.
        "INGESTION_UPLOAD_EXTENSIONS": ".pdf,.docx,.xlsx,.txt,.json,.xml,.png,.jpg,.jpeg,.md",
        "RATE_LIMIT_PER_MINUTE": "0",
    }
)

from tests.fakes import (  # noqa: E402
    FakeChatClient,
    FakeEmbeddingClient,
    FakeGraphStore,
    FakeVectorStore,
)


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def graph_store() -> FakeGraphStore:
    return FakeGraphStore()


@pytest.fixture
def embedding_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def chat_client() -> FakeChatClient:
    return FakeChatClient()


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.md").write_text("Annual leave policy: staff receive 25 days of leave per year.")
    (docs / "travel.txt").write_text("Travel policy: book flights through the travel portal.")
    (docs / "ignored.bin").write_bytes(b"\x00\x01")
    return tmp_path


@pytest.fixture
def ingestion_pipeline(embedding_client, vector_store, graph_store, chat_client, data_root):
    from app.rag.graph.extraction import GraphExtractor
    from app.rag.ingestion.chunking import ChunkingOptions
    from app.rag.ingestion.governance import GovernancePolicy
    from app.rag.ingestion.pipeline import IngestionPipeline, PipelineOptions, SourceConfig

    return IngestionPipeline(
        embedding_client,
        vector_store,
        sources=SourceConfig(local_root=str(data_root)),
        graph_store=graph_store,
        graph_extractor=GraphExtractor(chat_client),
        options=PipelineOptions(
            chunking=ChunkingOptions(chunk_size=50, chunk_overlap=10, min_chunk_tokens=5),
            # Test documents are short; production keeps the 50-character floor.
            governance=GovernancePolicy(min_text_chars=1),
            default_targets=("vector",),
        ),
    )


@pytest.fixture
def strategies(embedding_client, vector_store, graph_store, chat_client):
    from app.core.config import RagMode
    from app.rag.graph.extraction import GraphExtractor
    from app.rag.query.transforms import QueryTransformer
    from app.rag.retrieval.retriever import VectorRetriever
    from app.rag.strategies.agentic import AgenticRagStrategy
    from app.rag.strategies.graph import GraphRagStrategy
    from app.rag.strategies.hybrid_graph import HybridGraphRagStrategy
    from app.rag.strategies.semantic import SemanticRagStrategy

    def make_semantic(mode: RagMode) -> SemanticRagStrategy:
        return SemanticRagStrategy(
            VectorRetriever(embedding_client, vector_store),
            vector_store,
            transformer=QueryTransformer(chat_client),
            mode=mode,
        )

    semantic = make_semantic(RagMode.SEMANTIC)
    hybrid = make_semantic(RagMode.HYBRID)
    graph = GraphRagStrategy(graph_store, GraphExtractor(chat_client))
    agentic = AgenticRagStrategy(chat_client, {"vector": hybrid, "graph": graph})
    return {
        RagMode.SEMANTIC: semantic,
        RagMode.HYBRID: hybrid,
        RagMode.HYBRID_GRAPH: HybridGraphRagStrategy(hybrid, graph),
        RagMode.GRAPH: graph,
        RagMode.AGENTIC: agentic,
    }


@pytest.fixture
def retrieval_service(strategies, chat_client):
    from app.core.config import RagMode
    from app.rag.evaluation.judge import AnswerJudge
    from app.rag.evaluation.quality import QualityAssessor
    from app.rag.query.analysis import QueryAnalyzer
    from app.rag.safety import ContentSafety
    from app.rag.strategies.registry import RagStrategyRegistry
    from app.services.retrieval_service import RetrievalService

    return RetrievalService(
        RagStrategyRegistry(strategies, RagMode.SEMANTIC),
        chat_client,
        QualityAssessor(AnswerJudge(chat_client), judge_policy="request"),
        analyzer=QueryAnalyzer(chat_client, doc_types=["policy", "faq"]),
        safety=ContentSafety(mode="heuristic"),
    )


@pytest.fixture
def client(ingestion_pipeline, vector_store, graph_store, retrieval_service, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import deps
    from app.main import app
    from app.rag.vectorstore import factory
    from app.services.conversations import InMemoryConversationStore
    from app.services.ingestion_service import IngestionService

    monkeypatch.setattr(factory, "_vector_store_singleton", vector_store)
    ingestion_service = IngestionService(ingestion_pipeline, vector_store, graph_store)
    conversations = InMemoryConversationStore()
    monkeypatch.setattr(retrieval_service, "_conversations", conversations)
    app.dependency_overrides[deps.get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[deps.get_conversation_store] = lambda: conversations
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

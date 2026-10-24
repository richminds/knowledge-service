"""Dependency wiring: builds only what the enabled modes and configured services need.

Settings (a pydantic model) isn't hashable, so singletons are cached in module globals
rather than functools.lru_cache. Tests swap services with app.dependency_overrides.
"""

from typing import Annotated

from fastapi import Depends

from app.core.config import RagMode, Settings, get_settings
from app.rag.cache import AnswerCache
from app.rag.context.assembler import AssemblyOptions, ContextAssembler
from app.rag.evaluation.judge import AnswerJudge
from app.rag.evaluation.quality import QualityAssessor
from app.rag.generation.llm_client import ChatClient
from app.rag.generation.llm_gateway import get_llm_client, reset_llm_client
from app.rag.generation.llm_gateway_sdk import RemoteLLMClient
from app.rag.graph.extraction import GraphExtractor
from app.rag.graph.factory import get_graph_store
from app.rag.ingestion.embeddings import EmbeddingCache, EmbeddingClient
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.query.analysis import QueryAnalyzer
from app.rag.query.glossary import load_glossary
from app.rag.query.transforms import QueryTransformer
from app.rag.retrieval.reranker import CohereReranker, LlmReranker, Reranker, ScoreReranker
from app.rag.retrieval.retriever import VectorRetriever
from app.rag.safety import ContentSafety
from app.rag.strategies.agentic import AgenticRagStrategy
from app.rag.strategies.base import RagStrategy
from app.rag.strategies.graph import GraphRagStrategy
from app.rag.strategies.hybrid_graph import HybridGraphRagStrategy
from app.rag.strategies.registry import RagStrategyRegistry
from app.rag.strategies.semantic import SemanticOptions, SemanticRagStrategy
from app.rag.vectorstore.factory import get_vector_store
from app.services.conversations import (
    ConversationStore,
    InMemoryConversationStore,
    MongoConversationStore,
)
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

SettingsDep = Annotated[Settings, Depends(get_settings)]

_embedding_client: EmbeddingClient | None = None
_chat_client: ChatClient | None = None
_ingestion_service: IngestionService | None = None
_retrieval_service: RetrievalService | None = None
_conversations: ConversationStore | None = None
_cache: AnswerCache | None = None
_safety: ContentSafety | None = None


def get_gateway_client(settings: Settings) -> RemoteLLMClient:
    """The one pooled LLM gateway client (chat and embeddings)."""
    return get_llm_client(
        settings.llm_gateway_url,
        settings.api_gateway_url,
        settings.llm_gateway_api_key.get_secret_value(),
        settings.llm_timeout_seconds,
    )


def get_embedding_client(settings: SettingsDep) -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient(
            get_gateway_client(settings),
            batch_size=settings.embedding_batch_size,
            dimensions=settings.embedding_dimensions,
            caller=settings.llm_caller,
        )
    return _embedding_client


def get_chat_client(settings: SettingsDep) -> ChatClient:
    global _chat_client
    if _chat_client is None:
        _chat_client = ChatClient(
            get_gateway_client(settings),
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
            caller=settings.llm_caller,
            prompt_versions=settings.prompt_versions,
            organisation=settings.organisation_name,
        )
    return _chat_client


def get_conversation_store(settings: SettingsDep) -> ConversationStore:
    global _conversations
    if _conversations is None:
        if settings.conversation_store == "mongo":
            from app.core.mongo import get_mongo_database

            database = get_mongo_database(
                settings.mongodb_url.get_secret_value(), settings.mongodb_database
            )
            _conversations = MongoConversationStore(
                database[settings.mongodb_conversations_collection]
            )
        else:
            _conversations = InMemoryConversationStore()
    return _conversations


def get_answer_cache(settings: Settings) -> AnswerCache | None:
    global _cache
    if _cache is None and settings.answer_cache_enabled:
        _cache = AnswerCache(
            threshold=settings.answer_cache_similarity,
            ttl_seconds=settings.answer_cache_ttl_seconds,
            redis_url=settings.redis_url.get_secret_value(),
        )
    return _cache


def get_content_safety(settings: Settings) -> ContentSafety:
    global _safety
    if _safety is None:
        _safety = ContentSafety(
            mode=settings.content_safety_mode,
            block_output_pii=settings.block_output_pii,
        )
    return _safety


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "llm":
        return LlmReranker(get_chat_client(settings))
    if settings.reranker == "cohere":
        return CohereReranker(
            settings.reranker_endpoint,
            settings.reranker_api_key.get_secret_value(),
            settings.reranker_model,
        )
    return ScoreReranker(min_score=settings.retrieval_min_score)


def build_graph_extractor(settings: Settings) -> GraphExtractor:
    from app.rag.graph.ontology import load_ontology, load_resolver

    return GraphExtractor(
        get_chat_client(settings),
        ontology=load_ontology(settings.graph_ontology_path),
        resolver=load_resolver(settings.graph_aliases_path),
        gleaning_rounds=settings.graph_gleaning_rounds,
        extract_claims=settings.graph_extract_claims,
    )


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    from app.rag.ingestion import factory

    embedder = None
    if settings.vector_enabled:
        embedder = get_embedding_client(settings)
        if settings.embedding_cache_enabled:
            embedder = EmbeddingCache(
                embedder,
                model_key=f"llm-gateway@{settings.embedding_dimensions}",
                redis_url=settings.redis_url.get_secret_value(),
            )
    return IngestionPipeline(
        embedder,
        get_vector_store(settings) if settings.vector_enabled else None,
        graph_store=get_graph_store(settings) if settings.graph_enabled else None,
        graph_extractor=build_graph_extractor(settings) if settings.graph_enabled else None,
        connectors=factory.build_connectors(settings),
        parser=factory.build_parser(settings),
        manifest=factory.build_manifest(settings),
        zones=factory.build_zones(settings),
        enricher=factory.build_enricher(settings, get_chat_client(settings)),
        options=factory.build_options(settings),
    )


def build_strategy_registry(settings: Settings) -> RagStrategyRegistry:
    """Instantiate only the strategies in RAG_ENABLED_MODES (and what agentic depends on)."""
    enabled = set(settings.rag_enabled_modes)
    semantic: SemanticRagStrategy | None = None
    hybrid: SemanticRagStrategy | None = None
    graph: GraphRagStrategy | None = None
    graph_no_fallback: GraphRagStrategy | None = None
    transformer = QueryTransformer(get_chat_client(settings))

    if settings.vector_enabled:
        semantic_options = SemanticOptions(
            query_transform=settings.semantic_query_transform,
            multi_query_count=settings.semantic_multi_query_count,
            min_score=settings.retrieval_min_score,
            mmr=settings.semantic_mmr,
            mmr_lambda=settings.semantic_mmr_lambda,
            expand=settings.semantic_expand,
            recency_boost=settings.semantic_recency_boost,
            recency_half_life_days=settings.semantic_recency_half_life_days,
            hierarchical=settings.semantic_hierarchical,
            hierarchical_docs=settings.semantic_hierarchical_docs,
        )

        def make_semantic(mode: RagMode) -> SemanticRagStrategy:
            return SemanticRagStrategy(
                VectorRetriever(get_embedding_client(settings), get_vector_store(settings)),
                get_vector_store(settings),
                transformer=transformer,
                glossary=load_glossary(settings.glossary_path),
                options=semantic_options,
                mode=mode,
            )

        semantic = make_semantic(RagMode.SEMANTIC)
        # Hybrid (vector + BM25) also backs the graph fallback and the agent's vector tool.
        hybrid = make_semantic(RagMode.HYBRID)
    if settings.graph_enabled:
        from app.rag.graph.cypher import CypherPlanner, load_templates
        from app.rag.strategies.graph import GraphOptions

        vectors = get_vector_store(settings) if settings.vector_enabled else None
        graph_options = GraphOptions(
            mode=settings.graph_search_mode,
            max_hops=settings.graph_max_hops,
            max_entities=settings.graph_max_entities,
            max_relations=settings.graph_max_relations,
            per_entity_limit=settings.graph_per_entity_limit,
            min_weight=settings.graph_min_relation_weight,
            use_ppr=settings.graph_use_ppr,
            include_claims=settings.graph_include_claims,
            global_reports=settings.graph_global_reports,
            allow_freeform_cypher=settings.graph_allow_freeform_cypher,
            knee=settings.graph_knee_cutoff,
            knee_sensitivity=settings.retrieval_knee_sensitivity,
        )

        def make_graph(fallback: RagStrategy | None) -> GraphRagStrategy:
            return GraphRagStrategy(
                get_graph_store(settings),
                build_graph_extractor(settings),
                options=graph_options,
                llm=get_chat_client(settings),
                embedder=get_embedding_client(settings) if vectors is not None else None,
                vector_store=vectors,
                semantic_fallback=fallback,
                cypher=CypherPlanner(
                    get_chat_client(settings), load_templates(settings.graph_cypher_templates_path)
                ),
            )

        graph = make_graph(hybrid if settings.graph_semantic_fallback else None)
        # hybrid_graph runs the hybrid search itself, so its graph side needs no fallback.
        graph_no_fallback = make_graph(None)

    strategies: dict[RagMode, RagStrategy] = {}
    if RagMode.SEMANTIC in enabled and semantic:
        strategies[RagMode.SEMANTIC] = semantic
    if RagMode.HYBRID in enabled and hybrid:
        strategies[RagMode.HYBRID] = hybrid
    if RagMode.HYBRID_GRAPH in enabled and hybrid and graph_no_fallback:
        strategies[RagMode.HYBRID_GRAPH] = HybridGraphRagStrategy(hybrid, graph_no_fallback)
    if RagMode.GRAPH in enabled and graph:
        strategies[RagMode.GRAPH] = graph
    if RagMode.AGENTIC in enabled:
        from app.rag.strategies.agentic import AgentLimits, build_checkpointer

        strategies[RagMode.AGENTIC] = AgenticRagStrategy(
            get_chat_client(settings),
            tools=build_agent_tools(settings, hybrid, graph),
            limits=AgentLimits(
                max_iterations=settings.agentic_max_iterations,
                max_tool_calls=settings.agentic_max_tool_calls,
                max_tokens=settings.agentic_max_tokens,
                step_timeout_seconds=settings.agentic_step_timeout_seconds,
                max_seconds=settings.agentic_max_seconds,
                min_relevant=settings.agentic_min_relevant,
                max_reflections=settings.agentic_max_reflections,
                allow_clarify=settings.agentic_allow_clarify,
                generate_and_check=settings.agentic_generate_and_check,
                review_on_failed_check=settings.agentic_review_on_failed_check,
            ),
            checkpointer=build_checkpointer(
                settings.agentic_checkpointer,
                settings.agentic_checkpoint_dsn.get_secret_value()
                or settings.mongodb_url.get_secret_value(),
            ),
        )
    return RagStrategyRegistry(
        strategies, settings.rag_mode, settings.rag_allow_mode_override, settings.rag_router_map
    )


def build_agent_tools(settings: Settings, semantic, graph) -> list:
    """D2: the tool set configured by AGENTIC_TOOLS (only tools whose backend is enabled)."""
    from app.rag.agents import tools as t

    wanted = set(settings.agentic_tools)
    tools = []
    if semantic is not None:
        if "vector" in wanted:
            tools.append(
                t.strategy_tool(
                    "semantic_search",
                    "Meaning-based search over document passages: policies, procedures, "
                    "explanations.",
                    semantic,
                    {"vector"},
                )
            )
        if "keyword" in wanted:
            tools.append(
                t.keyword_tool(
                    VectorRetriever(get_embedding_client(settings), get_vector_store(settings))
                )
            )
        if "fetch_document" in wanted:
            tools.append(t.fetch_document_tool(get_vector_store(settings)))
    if graph is not None:
        if "graph" in wanted:
            tools.append(
                t.graph_mode_tool(
                    "graph_local_search",
                    "Knowledge graph around named entities: who manages/owns/depends on what, "
                    "multi-hop links.",
                    graph,
                    "local",
                )
            )
        if "graph_global" in wanted:
            tools.append(
                t.graph_mode_tool(
                    "graph_global_search",
                    "Summaries of topic clusters for broad questions: main themes, overall "
                    "picture.",
                    graph,
                    "global",
                )
            )
        if "graph_query" in wanted:
            tools.append(
                t.graph_mode_tool(
                    "graph_query",
                    "Precise structured lookups in the graph (lists, who-manages-X style "
                    "questions).",
                    graph,
                    "cypher",
                )
            )
    if "calculator" in wanted:
        tools.append(t.calculator_tool())
    if "date_math" in wanted:
        tools.append(t.date_math_tool())
    if (
        "web" in wanted
        and settings.web_search_endpoint
        and settings.web_search_api_key.get_secret_value()
    ):
        tools.append(
            t.web_search_tool(
                settings.web_search_endpoint, settings.web_search_api_key.get_secret_value()
            )
        )
    return tools


def get_ingestion_service(settings: SettingsDep) -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService(
            build_ingestion_pipeline(settings),
            get_vector_store(settings) if settings.vector_enabled else None,
            get_graph_store(settings) if settings.graph_enabled else None,
            cache=get_answer_cache(settings),
        )
    return _ingestion_service


def get_retrieval_service(settings: SettingsDep) -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        chat_client = get_chat_client(settings)
        assessor = QualityAssessor(
            judge=(
                AnswerJudge(chat_client, settings.quality_judge_max_passage_chars)
                if settings.quality_judge != "off"
                else None
            ),
            judge_policy=settings.quality_judge,
            background_sample_rate=settings.quality_judge_sample_rate,
        )
        cache = get_answer_cache(settings)
        _retrieval_service = RetrievalService(
            build_strategy_registry(settings),
            chat_client,
            assessor,
            assembler=ContextAssembler(
                build_reranker(settings),
                AssemblyOptions(
                    top_n=settings.retrieval_rerank_top_n,
                    min_relative_score=settings.context_min_relative_score,
                    max_context_tokens=settings.context_max_tokens,
                    reorder=settings.context_reorder,
                    knee=settings.retrieval_knee_cutoff,
                    knee_sensitivity=settings.retrieval_knee_sensitivity,
                ),
            ),
            analyzer=QueryAnalyzer(
                chat_client,
                doc_types=settings.doc_types,
                assistant_scope=settings.assistant_scope,
                enabled=settings.query_analysis_enabled,
            ),
            safety=get_content_safety(settings),
            glossary=load_glossary(settings.glossary_path),
            cache=cache,
            conversations=get_conversation_store(settings),
            embedder=(
                VectorRetriever(get_embedding_client(settings), get_vector_store(settings))
                if cache is not None and settings.vector_enabled
                else None
            ),
            max_history_turns=settings.conversation_max_turns,
            log_question_text=settings.log_question_text,
            escalation=settings.rag_escalation_enabled,
            max_escalations=settings.rag_max_escalations,
            vector_backends=settings.active_vector_providers if settings.vector_enabled else (),
            default_vector_backend=settings.vector_store_provider,
        )
    return _retrieval_service


def reset() -> None:
    """Drop cached singletons (ablation runs rebuild services with different settings)."""
    global _embedding_client, _chat_client, _ingestion_service, _retrieval_service
    global _conversations, _cache, _safety
    _embedding_client = _chat_client = _ingestion_service = _retrieval_service = None
    _conversations = _cache = _safety = None
    reset_llm_client()
    from app.rag.graph import factory as graph_factory
    from app.rag.vectorstore import factory as vector_factory

    vector_factory._vector_store_singleton = None
    graph_factory._graph_store_singleton = None


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
ConversationStoreDep = Annotated[ConversationStore, Depends(get_conversation_store)]

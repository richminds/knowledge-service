# 0001. FastAPI + LangGraph, no all-in-one RAG framework (C-016)

**Context.** The system needs three retrieval strategies, a router and an agent loop, and every step must be observable and permission-aware. All-in-one frameworks such as LlamaIndex and Haystack speed up a prototype but hide ACL filtering, tracing and prompt control behind abstractions.

**Decision.**
- The API is built on FastAPI (async, Pydantic models, OpenAPI).
- The agent workflow uses LangGraph. It gives explicit state, conditional edges and checkpointing.
- Retrieval, fusion, generation and evaluation are plain Python modules behind small interfaces: `RagStrategy`, `VectorStore` and `GraphStore`.

**Consequences.**
- There is more code to own. In return, each stage is testable with fakes (`tests/fakes.py`) and traced as its own span.
- Swapping a store means implementing one interface.
- Upgrades to LangGraph only affect `rag/strategies/agentic.py`.

# 0004. Three RAG modes behind a flag and a router (I-01, I-05)

**Context.** We need to validate semantic, graph and agentic RAG side by side on the same data, and then serve the cheapest one that is good enough.

**Decision.**
- `RAG_MODE` is one of `semantic`, `graph`, `agentic` or `auto`.
- `RAG_ENABLED_MODES` turns each path on or off.
- `RAG_ALLOW_MODE_OVERRIDE` lets QA pick a mode per request.
- `auto` routes by question complexity (`RAG_ROUTER_MAP`). It escalates to a more capable mode when confidence is low (`RAG_ESCALATION_ENABLED`).
- Every path returns the same response schema: answer, citations, confidence and trace.

**Consequences.**
- QA can compare modes with `scripts/run_eval.py --rag-mode all`. The workbook shows the traffic split, quality and cost per mode.
- Prod can switch modes with a parameter change, without a code change.

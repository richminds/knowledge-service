# Agentic RAG for Beginners: How It Works and Why Use It

This guide explains, in plain language, how the **agentic** retrieval mode works in this project and what it gives you compared with the other modes.

It builds on [graph-rag-workflow.md](./graph-rag-workflow.md). If words like *chunk*, *entity* or *vector store* are new, read the glossary in section 2 of that guide first.

**Contents**

1. [What is agentic RAG?](#1-what-is-agentic-rag)
2. [How it differs from the other modes](#2-how-it-differs-from-the-other-modes)
3. [The workflow at a glance](#3-the-workflow-at-a-glance)
4. [Each step explained](#4-each-step-explained)
5. [The tools the agent can use](#5-the-tools-the-agent-can-use)
6. [A worked example](#6-a-worked-example)
7. [Safety limits: how the agent is kept under control](#7-safety-limits-how-the-agent-is-kept-under-control)
8. [Benefits](#8-benefits)
9. [Trade-offs](#9-trade-offs)
10. [When agentic RAG is used](#10-when-agentic-rag-is-used)
11. [Settings you will touch most](#11-settings-you-will-touch-most)
12. [Where to find things in the code](#12-where-to-find-things-in-the-code)

---

## 1. What is agentic RAG?

In normal RAG, the system searches **once**, takes the top results and asks the LLM to answer. If the first search misses something, the answer suffers.

In **agentic RAG**, an LLM acts like a **researcher** instead of a one-shot search box. It can:

- **Plan**: split a hard question into smaller questions.
- **Choose tools**: pick the right search for each part (meaning search, keyword search, graph search, a calculator, ...).
- **Check**: decide whether the evidence it found is good enough.
- **Retry**: rewrite the search, or follow a lead ("this policy mentions a system, let me look that system up").
- **Verify**: check its own draft answer against the evidence before returning it.

> **Analogy:** asking a colleague a tricky question.
> - Normal RAG = they run one search, read the top results and reply.
> - Agentic RAG = they break your question into parts, search in different places, notice when something is missing, search again, then double-check their answer before replying.

The loop of *think, search, check, retry* is what makes it "agentic".

---

## 2. How it differs from the other modes

| Mode | Searches | LLM calls | Best for |
|---|---|---|---|
| `semantic` | Vector search once | 1 (answer) plus query analysis | Simple facts |
| `hybrid` | Vector plus keyword search once | Same | Simple facts, exact names and codes |
| `graph` | Graph walk (one search mode) | Entity extraction plus answer | Relationships, big-picture questions |
| `hybrid_graph` | Hybrid and graph in parallel, fused with RRF | Same as graph | Questions that benefit from both |
| `agentic` | **Several**, chosen and repeated by the agent | **Many** (route, plan, grade, refine, answer, check) | Multi-step, comparison or analysis questions |

The first four modes follow a **fixed** path. Agentic RAG decides its path **at question time**, based on what it finds.

Agentic RAG does not replace the other modes. It **uses** them: the semantic, keyword and graph searches are available to the agent as tools.

---

## 3. The workflow at a glance

The agent is a state machine built with **LangGraph** ([agentic.py](../src/app/rag/strategies/agentic.py)). Each box is a node, and the arrows are decisions.

```mermaid
flowchart TD
    START([User question]) --> ROUTE["1. Route<br/>pick tools, judge complexity"]
    ROUTE -->|needs clarification<br/>(off by default)| ASK([Ask user a clarifying question])
    ROUTE --> PLAN["2. Plan<br/>split into sub-questions"]
    PLAN --> RETRIEVE["3. Retrieve<br/>run tools in parallel"]
    RETRIEVE --> GRADE["4. Grade<br/>keep only relevant passages"]
    GRADE --> ENOUGH{"5. Enough evidence<br/>or out of budget?"}
    ENOUGH -->|no, budget left| REFINE["6. Refine<br/>rewrite query or follow a lead"]
    REFINE --> RETRIEVE
    ENOUGH -->|yes| GEN["7. Generate<br/>draft the answer"]
    GEN --> CHECK["8. Check<br/>grounded? answers the question?"]
    CHECK -->|failed, budget left| REFINE
    CHECK -->|passed, or out of budget| END([Answer + citations])
    ENOUGH -->|no evidence at all<br/>and out of budget| END
```

---

## 4. Each step explained

| # | Step | What happens | Where the LLM is used |
|---|---|---|---|
| 1 | **Route** | One cheap LLM call looks at the question and the list of tools. It chooses which tools to use, whether the question is `simple` or `multi_step`, and (only if enabled) whether to ask the user to clarify. If the LLM fails, every search tool is used | Yes (`agent_route`) |
| 2 | **Plan** | For `multi_step` questions, the question is split into at most 4 ordered sub-questions. Simple questions skip this | Yes (`agent_plan`) |
| 3 | **Retrieve** | Every chosen tool runs for every sub-question, **in parallel**, each with a timeout. Results are merged with RRF. A search already done is never repeated. If a tool errors or times out, another search tool is tried | No |
| 4 | **Grade** | The LLM reads the retrieved passages and keeps only those that help answer the question. If grading fails, everything is kept | Yes (`agent_grade`) |
| 5 | **Enough?** | A rule, not an LLM: is there enough relevant evidence (default 2 passages)? Is any budget left? Enough evidence, or no budget, leads to generating an answer | No |
| 6 | **Refine** | If evidence was found but is not enough, the LLM writes the **next hop** query (for example, look up an entity mentioned in the evidence). If nothing relevant was found, it **rewrites** the query with synonyms, expanded acronyms or more specific names | Yes (`agent_hop` / `agent_rewrite`) |
| 7 | **Generate** | The answer is drafted from the evidence using the same generation stack as the other modes, with numbered citations | Yes |
| 8 | **Check** | The LLM checks the draft: is every claim **supported** by the passages, and does it **answer the question**? | Yes (`agent_check`) |

After step 8:

- **Passed:** the answer is returned.
- **Failed and budget left:** the checker's note about what is missing becomes the next query, and the agent goes back to **Refine** (this is called *reflection*, by default at most once).
- **Failed and no budget:** the answer is returned, but flagged for **human review** (`review_required`).

If the agent cannot find any evidence at all, it ends without an answer, and the service replies with the normal "not found" message.

---

## 5. The tools the agent can use

Each tool has a name, a description that tells the LLM **when** to use it, and a parameter schema ([tools.py](../src/app/rag/agents/tools.py)). Tools are **read-only**, and each one enforces the user's permissions itself, so the agent cannot see anything the user cannot.

| Tool | What it does | Typical use |
|---|---|---|
| `semantic_search` | Meaning-based search over chunks | General questions |
| `keyword_search` | Exact-term search (BM25) | Document numbers, codes, acronyms, names |
| `graph_local_search` | Entities, relations and multi-hop facts | "Who owns the system X depends on?" |
| `graph_global_search` | Community reports across the whole collection | "What are the main themes?" |
| `graph_query` | Approved or validated Cypher queries | Counts and lists |
| `fetch_document` | The full text of a document already found | When one passage is not enough |
| `calculator` | Safe arithmetic (numbers and operators only) | Totals, percentages, differences |
| `date_math` | Days between dates, or date plus N days | Deadlines |
| `web_search` | Optional external search | Off by default; needs endpoint and key, and your data policy must allow it |

Which tools are active is set by `AGENTIC_TOOLS`. Utility tools (`calculator`, `date_math`) and `web_search` are not used as automatic fallbacks, only when the LLM picks them.

---

## 6. A worked example

**Question:** "Which team owns the system that Acme Corp depends on, and when was the related policy last updated?"

This has two parts, so the router marks it `multi_step`.

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Tools
    U->>A: Question
    A->>A: 1. Route: tools = graph_local_search, semantic_search; multi_step
    A->>A: 2. Plan: sub-questions
    Note over A: (a) Which system does Acme Corp depend on and who owns it?<br/>(b) When was the related policy last updated?
    par Round 1 (parallel)
        A->>T: graph_local_search(a)
        A->>T: semantic_search(a)
        A->>T: graph_local_search(b)
        A->>T: semantic_search(b)
    end
    T-->>A: Candidates, merged with RRF
    A->>A: 4. Grade: keeps facts about Billing System and IT Department, drops the rest
    A->>A: 5. Enough? Ownership found, policy date missing
    A->>A: 6. Refine: next hop = "Billing System policy last updated"
    A->>T: Round 2 (semantic_search, keyword_search)
    T-->>A: Passage with the policy date
    A->>A: 4. Grade, 5. Enough? Yes
    A->>A: 7. Generate the draft answer
    A->>A: 8. Check: grounded and complete
    A-->>U: Answer with citations
```

What this shows:

- The question was **split**, and each part used the best tool.
- The first round answered only half. The agent **noticed** and ran a second, targeted search.
- A plain `hybrid` search would likely have returned passages about only one of the two parts.
- A final **check** confirmed each claim is backed by a retrieved passage.

---

## 7. Safety limits: how the agent is kept under control

An agent that can loop needs hard limits, so it cannot run forever or spend without bound. These come from `AgentLimits` and the `AGENTIC_*` settings.

| Limit | Default | What it stops |
|---|---|---|
| Retrieve rounds | 3 | Endless searching |
| Tool calls per run | 12 | Too many searches |
| Token budget per run | 60,000 | Runaway LLM cost |
| Time per tool call | 20 seconds | A slow tool blocking the run |
| Total time | 90 seconds | A run that never ends |
| Sub-questions | 4 | Over-splitting |
| Reflections | 1 | Endless check-and-retry loops |

Other protections:

- **No repeated searches:** the agent keeps a list of searches done (tool plus query) and skips repeats.
- **Fallback on failure:** if a tool times out or errors, another search tool is tried.
- **Graceful stop:** when a limit is hit, the agent answers with what it has instead of failing.
- **Read-only tools:** nothing the agent calls changes data.
- **Permissions in the tools:** the tools, not the agent, apply the user's access rules.
- **Passage safety:** retrieved passages still go through the normal prompt-injection check afterwards. If any passage is dropped, the agent's pre-written answer is discarded and the answer is generated again from the remaining passages.
- **Review flag:** if the final self-check fails, the response is marked `review_required` (`AGENTIC_REVIEW_ON_FAILED_CHECK`).

---

## 8. Benefits

| Benefit | Why |
|---|---|
| **Handles multi-step questions** | Breaking a question into sub-questions covers every part, where a single search often covers only one |
| **Uses the right search for each job** | Keyword search for codes, graph search for relationships, global search for themes, all in one run |
| **Recovers from a weak first search** | Grading shows the evidence is thin, so the agent rewrites the query or follows a lead instead of answering from poor results |
| **Follows chains of facts** | The "next hop" step finds information that is only reachable after reading the first results |
| **Fewer unsupported answers** | The grounding check compares the draft to the evidence and retries or flags it |
| **Does simple calculations and date maths correctly** | The calculator and date tools replace the LLM's guesswork with exact results |
| **Explainable** | Every step is traced, and citations carry the tool that found them, which gives a clear evidence trail |
| **Controlled** | Hard limits on rounds, calls, tokens and time keep cost and latency bounded |
| **Resilient** | Timeouts, tool fallbacks and graceful stops mean one failing tool does not break the answer |
| **Extensible** | A new tool (for example, a SQL lookup) can be added with a name, description and function |

---

## 9. Trade-offs

| Cost | Detail |
|---|---|
| **Slower** | Several LLM calls and several search rounds, up to the 90-second limit |
| **More expensive** | Routing, planning, grading, refining and checking each use LLM tokens (capped at 60,000 per run by default) |
| **Harder to predict** | The path depends on what the agent finds, so two runs can differ slightly |
| **More to tune** | Tools, limits and prompts all affect results |
| **Overkill for simple questions** | A fact lookup is answered faster and cheaper by `hybrid` |

For these reasons, agentic RAG is used where the extra effort pays off, not for every question.

---

## 10. When agentic RAG is used

**By the router.** With `RAG_MODE=auto`, the question's complexity decides the mode ([registry.py](../src/app/rag/strategies/registry.py)):

| Complexity | Default mode |
|---|---|
| `simple` | `hybrid` |
| `relationship` | `graph` |
| `global` | `graph` |
| `multi_step` | **`agentic`** |

You can change this with `RAG_ROUTER_MAP`.

**By choice.** A request can set `rag_mode` to `agentic` (when `RAG_ALLOW_MODE_OVERRIDE` is on), or you can make it the default with `RAG_MODE=agentic`.

**By escalation.** With `RAG_ESCALATION_ENABLED=true`, a **low-confidence** answer from another mode is retried with a stronger one. Agentic is the last step in every chain:

- `semantic` goes to `hybrid`, then `hybrid_graph`, then `graph`, then `agentic`.
- `hybrid` goes to `hybrid_graph`, then `graph`, then `agentic`.
- `graph` goes to `hybrid_graph`, then `agentic`.
- `hybrid_graph` goes to `agentic`.

The mode must also be listed in `RAG_ENABLED_MODES`. If it is not, the router falls back to the closest enabled mode.

---

## 11. Settings you will touch most

Set in the `.env` file (see [.env.example](../.env.example) for all options).

| Setting | Default | What it changes |
|---|---|---|
| `RAG_MODE` | `hybrid` | Default mode (`agentic` or `auto` to use the agent) |
| `RAG_ENABLED_MODES` | all five | Modes that may be used |
| `RAG_ROUTER_MAP` | see `.env.example` | Which question type goes to which mode |
| `RAG_ESCALATION_ENABLED` | `false` | Retry low-confidence answers with a stronger mode |
| `AGENTIC_TOOLS` | `vector,keyword,fetch_document,graph,graph_global,calculator,date_math` | Tools the agent may use |
| `AGENTIC_MAX_ITERATIONS` | `3` | Retrieve rounds |
| `AGENTIC_MIN_RELEVANT` | `2` | Relevant passages needed before answering |
| `AGENTIC_MAX_TOOL_CALLS` | `12` | Tool calls per run |
| `AGENTIC_MAX_TOKENS` | `60000` | Token budget per run |
| `AGENTIC_STEP_TIMEOUT_SECONDS` | `20.0` | Timeout per tool call |
| `AGENTIC_MAX_SECONDS` | `90.0` | Total time per run |
| `AGENTIC_MAX_REFLECTIONS` | `1` | Check-and-retry loops |
| `AGENTIC_ALLOW_CLARIFY` | `false` | Let the agent ask the user a clarifying question |
| `AGENTIC_GENERATE_AND_CHECK` | `true` | Draft and self-check the answer inside the agent |
| `AGENTIC_REVIEW_ON_FAILED_CHECK` | `true` | Flag answers that failed the final check |
| `AGENTIC_CHECKPOINTER` | `none` | Save agent state between steps (`memory` or `mongo`) |
| `WEB_SEARCH_ENDPOINT` / `WEB_SEARCH_API_KEY` | empty | Enable the optional web search tool |

---

## 12. Where to find things in the code

| I want to understand... | Open |
|---|---|
| The agent loop and its steps | [strategies/agentic.py](../src/app/rag/strategies/agentic.py) |
| The tools | [agents/tools.py](../src/app/rag/agents/tools.py) |
| How the router picks a mode and escalation | [strategies/registry.py](../src/app/rag/strategies/registry.py) |
| How the answer is used after retrieval | [services/retrieval_service.py](../src/app/services/retrieval_service.py) |
| The graph searches the agent calls | [strategies/graph.py](../src/app/rag/strategies/graph.py) |

---

## Quick recap

1. **Agentic RAG** lets an LLM plan, search with several tools, grade the evidence, retry, and check its own answer.
2. The loop is **route, plan, retrieve, grade, refine, generate, check**, repeated within strict limits.
3. Its tools include semantic, keyword and graph search, document fetch, a calculator, date maths and optional web search.
4. The **benefits** are better answers to multi-step questions, recovery from weak searches, fewer unsupported answers, and a clear evidence trail.
5. The **cost** is more time and more LLM calls, so it is used for `multi_step` questions or as the last escalation step, and simple questions stay on `hybrid`.

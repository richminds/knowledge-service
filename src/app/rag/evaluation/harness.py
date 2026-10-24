"""Shared evaluation harness (A15, B4, C7, D6, I-02, I-07).

Golden set (JSONL, C-203/C-205), one case per line:
  {"id": "q1", "question": "...", "collection": "default",
   "question_type": "factoid|multi_hop|relationship|global|analytical|unanswerable",
   "expected_sources": ["policies/leave.pdf"],     # retrieval relevance by source suffix
   "relevant_chunk_ids": [],                         # or exact chunk ids
   "expected_answer": "...",                         # optional reference -> correctness
   "expected_tools": ["graph_local_search"],         # optional, agentic tool selection
   "expected_mode": "graph"}                         # optional, else from question_type

Metrics per RAG mode:
  retrieval (C-206): recall@k, precision@k, MRR, nDCG@k
  generation (C-207): faithfulness (judge groundedness), answer relevance, correctness
  citations (C-208): citation precision / recall
  cost & latency (C-210): tokens and ms per query, and ms spent in vector-store searches
  routing (I-02): router accuracy against the expected mode
  agentic (A-46..A-49): tool-selection accuracy, steps, redundant calls, loop rate
Results can be compared with a baseline (C-212), checked against thresholds (C-213, the CI
gate) and appended to a history file (C-215). Runs on a named vector store are labelled
"<mode>@<store>" (e.g. "semantic@mongo") and use the mode's thresholds.
"""

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from app.core.config import RagMode
from app.core.exceptions import GenerationError
from app.models.schemas import ChatRequest, ChatResponse

TYPE_TO_MODE = {
    "factoid": "hybrid",
    "unanswerable": "hybrid",
    "relationship": "graph",
    "multi_hop": "graph",
    "global": "graph",
    "analytical": "agentic",
    "multi_step": "agentic",
}
# Trace steps that query the vector store (embedding + search), timed as retrieval_ms.
VECTOR_SEARCH_STEPS = frozenset({"vector_search", "document_search", "vector_seed"})
CORRECTNESS_PROMPT = (
    "Compare a generated answer with the reference answer for the question. Score correctness "
    "1-5 (5 = same facts, nothing wrong; 3 = partially correct; 1 = wrong or missing). For an "
    "unanswerable question, a reply saying the information is not available is correct. "
    'Return JSON: {"correctness": int, "reason": str}.'
)


@dataclass
class GoldenCase:
    question: str
    id: str = ""
    collection: str = "default"
    question_type: str = "factoid"
    expected_sources: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_answer: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_mode: str = ""

    @property
    def mode(self) -> str:
        return self.expected_mode or TYPE_TO_MODE.get(self.question_type, "semantic")

    def is_relevant(self, source: str, chunk_id: str) -> bool:
        if chunk_id in self.relevant_chunk_ids:
            return True
        normalised = source.replace("\\", "/")
        return any(normalised.endswith(s) for s in self.expected_sources)


def load_golden(path: str | Path) -> list[GoldenCase]:
    cases = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            data = json.loads(line)
            known = {k: v for k, v in data.items() if k in GoldenCase.__dataclass_fields__}
            case = GoldenCase(**known)
            case.id = case.id or f"case-{i}"
            cases.append(case)
    return cases


# -- retrieval metrics (C-206) --------------------------------------------------------------------
def retrieval_metrics(relevance: list[bool], total_relevant: int, k: int) -> dict[str, float]:
    top = relevance[:k]
    hits = sum(top)
    first = next((i for i, r in enumerate(top, start=1) if r), None)
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(top, start=1) if r)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(total_relevant, k) + 1))
    return {
        "recall": hits / total_relevant if total_relevant else 0.0,
        "precision": hits / len(top) if top else 0.0,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
        "hit": float(hits > 0),
    }


def citation_metrics(case: GoldenCase, response: ChatResponse) -> dict[str, float]:
    """C-208: are the cited passages relevant, and were the relevant ones cited?"""
    cited = [c for c in response.citations if c.cited]
    relevant_cited = [c for c in cited if case.is_relevant(c.source, c.chunk_id)]
    expected = len(case.expected_sources) or len(case.relevant_chunk_ids)
    covered = {
        s
        for s in case.expected_sources
        if any(c.source.replace("\\", "/").endswith(s) for c in cited)
    }
    covered |= {cid for cid in case.relevant_chunk_ids if any(c.chunk_id == cid for c in cited)}
    return {
        "citation_precision": len(relevant_cited) / len(cited) if cited else 0.0,
        "citation_recall": len(covered) / expected if expected else 0.0,
    }


def agent_metrics(case: GoldenCase, response: ChatResponse) -> dict[str, float]:
    """A-46..A-48 from the trajectory in the trace."""
    trace = response.trace or []
    retrieves = [s for s in trace if s.step == "retrieve"]
    tools = [t for s in retrieves for t in s.detail.get("tools", [])]
    queries = [q for s in retrieves for q in s.detail.get("queries", [])]
    pairs = list(zip(tools, queries, strict=False))
    metrics = {
        "agent_steps": float(len(trace)),
        "agent_tool_calls": float(len(tools)),
        "agent_redundant_calls": float(len(pairs) - len(set(pairs))),
        "agent_looped": float(len(retrieves) > 1),
    }
    if case.expected_tools:
        metrics["tool_selection"] = float(set(case.expected_tools) <= set(tools))
    return metrics


async def score_correctness(llm, case: GoldenCase, answer: str) -> float | None:
    """Correctness 1-5 against the reference answer (C-207), or None without one."""
    if llm is None or not case.expected_answer:
        return None
    try:
        data = await llm.complete_json(
            CORRECTNESS_PROMPT,
            f"Question: {case.question}\nReference: {case.expected_answer}\nGenerated: {answer}",
            purpose="correctness_judge",
        )
        return float(min(5, max(1, int(data.get("correctness", 1)))))
    except (GenerationError, TypeError, ValueError):
        return None


def response_metrics(
    case: GoldenCase,
    response: ChatResponse,
    top_k: int,
    mode: RagMode | None,
    latency_ms: float,
) -> dict[str, float]:
    """Every metric one answer yields, except correctness (which needs an LLM call)."""
    relevance = [case.is_relevant(c.source, c.chunk_id) for c in response.citations]
    total = len(case.expected_sources) or len(case.relevant_chunk_ids) or int(any(relevance))
    metrics = retrieval_metrics(relevance, total, top_k) if total else {}
    metrics.update(citation_metrics(case, response) if total else {})
    metrics["latency_ms"] = latency_ms
    searches = [s for s in response.trace or [] if s.step in VECTOR_SEARCH_STEPS]
    if searches:
        metrics["retrieval_ms"] = sum(s.duration_ms for s in searches)
    metrics["tokens"] = float(response.usage.total_tokens if response.usage else 0)
    quality = response.quality
    if quality is not None:
        metrics["citation_coverage"] = quality.signals.citation_coverage
        metrics["answered"] = float(quality.signals.answered)
        if quality.judge is not None:
            metrics["faithfulness"] = float(quality.judge.groundedness)
            metrics["answer_relevance"] = float(quality.judge.answer_relevance)
            metrics["context_relevance"] = float(quality.judge.context_relevance)
    if case.question_type == "unanswerable":
        metrics["abstained"] = float(
            response.answer_type == "not_found" or not (quality and quality.signals.answered)
        )
    if mode in (None, RagMode.AUTO):
        metrics["routing_accuracy"] = float(str(response.rag_mode) == case.mode)  # I-02
    if response.rag_mode == RagMode.AGENTIC:
        metrics.update(agent_metrics(case, response))
    return metrics


@dataclass
class CaseResult:
    case_id: str
    question_type: str
    mode: str
    metrics: dict[str, float]
    answer: str = ""


class EvalRunner:
    def __init__(self, service, llm=None, top_k: int = 5) -> None:
        self._service = service
        self._llm = llm
        self._top_k = top_k

    async def _correctness(self, case: GoldenCase, answer: str) -> float | None:
        return await score_correctness(self._llm, case, answer)

    async def run_case(
        self,
        case: GoldenCase,
        mode: RagMode | None,
        judge: bool,
        vector_store: str | None = None,
    ) -> CaseResult:
        started = time.perf_counter()
        response: ChatResponse = await self._service.chat(
            ChatRequest(
                query=case.question,
                collection=case.collection,
                rag_mode=mode,
                vector_store=vector_store,
                evaluate=judge,
                include_trace=True,
            ),
            top_k=self._top_k,
            # A store comparison must hit each store, not an answer cached from the other.
            **({"use_cache": False, "force_mode": True} if vector_store else {}),
        )
        latency = (time.perf_counter() - started) * 1000
        metrics = response_metrics(case, response, self._top_k, mode, latency)
        correctness = await self._correctness(case, response.answer)
        if correctness is not None:
            metrics["correctness"] = correctness
        return CaseResult(
            case.id, case.question_type, str(response.rag_mode), metrics, response.answer
        )

    async def run(
        self,
        cases: list[GoldenCase],
        modes: list[RagMode | None],
        judge: bool = False,
        vector_stores: list[str | None] | None = None,
    ) -> dict:
        """Every case in every mode x vector store (None = the deployment default store)."""
        results: dict[str, list[CaseResult]] = {}
        for mode in modes:
            for store in vector_stores or [None]:
                label = str(mode) if mode is not None else "default"
                label += f"@{store}" if store else ""
                results[label] = [
                    await self.run_case(case, mode, judge, store) for case in cases
                ]
        return {
            "summary": {label: summarise(rows) for label, rows in results.items()},
            "by_type": {label: summarise_by_type(rows) for label, rows in results.items()},
            "cases": {label: [asdict(r) for r in rows] for label, rows in results.items()},
        }


def summarise(rows: list[CaseResult]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r.metrics})
    return {
        k: round(
            sum(r.metrics[k] for r in rows if k in r.metrics)
            / max(1, sum(k in r.metrics for r in rows)),
            4,
        )
        for k in keys
    } | {"cases": float(len(rows))}


def summarise_by_type(rows: list[CaseResult]) -> dict[str, dict[str, float]]:
    """B4/C7/D6 scorecards: each approach scored on the question types it targets."""
    types = sorted({r.question_type for r in rows})
    return {t: summarise([r for r in rows if r.question_type == t]) for t in types}


# -- gate, baseline, history (C-212, C-213, C-215) -------------------------------------------------
def check_thresholds(report: dict, thresholds_path: str | Path) -> list[str]:
    """evals/thresholds.yaml: {"all": {metric: min}, "<mode>": {metric: min}}; *_max for maxima.
    A "<mode>@<store>" run is held to the rules of <mode>."""
    path = Path(thresholds_path)
    if not path.is_file():
        return []
    rules = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    violations = []
    for mode, summary in report["summary"].items():
        mode_rules = rules.get(mode.split("@", 1)[0]) or {}
        for metric, limit in {**(rules.get("all") or {}), **mode_rules}.items():
            name, is_max = (metric[:-4], True) if metric.endswith("_max") else (metric, False)
            value = summary.get(name)
            if value is None:
                continue
            if (is_max and value > limit) or (not is_max and value < limit):
                violations.append(f"{mode}.{name}={value} {'>' if is_max else '<'} {limit}")
    return violations


def compare_baseline(report: dict, baseline_path: str | Path, tolerance: float = 0.02) -> list[str]:
    path = Path(baseline_path)
    if not path.is_file():
        return []
    baseline = json.loads(path.read_text(encoding="utf-8"))
    regressions = []
    lower_is_better = {
        "latency_ms", "retrieval_ms", "tokens", "agent_redundant_calls", "agent_steps"
    }  # fmt: skip
    for mode, summary in report["summary"].items():
        for metric, value in summary.items():
            before = (baseline.get("summary", {}).get(mode) or {}).get(metric)
            if before is None or metric == "cases":
                continue
            worse = (
                value > before * (1 + tolerance)
                if metric in lower_is_better
                else value < before - tolerance
            )
            if worse:
                regressions.append(f"{mode}.{metric}: {before} -> {value}")
    return regressions


def git_sha() -> str:
    sha = os.environ.get("BUILD_SOURCEVERSION", "")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()  # noqa: S603, S607
    except OSError:
        return ""


def append_history(report: dict, history_path: str | Path, context: dict | None = None) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(),
        "commit": git_sha(),
        **(context or {}),
        "summary": report["summary"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

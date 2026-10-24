"""Evaluation runner (A15) over the golden set, per RAG mode.

Usage:
    python scripts/run_eval.py                                  # default RAG_MODE
    python scripts/run_eval.py --rag-mode all --judge           # every mode + LLM judge
    python scripts/run_eval.py --rag-mode auto                  # routing accuracy (I-02)
    python scripts/run_eval.py --vector-store all --judge       # every configured store
    python scripts/run_eval.py --rag-mode semantic --vector-store mongo
    python scripts/run_eval.py --gate                           # thresholds + baseline (CI, C-213)
    python scripts/run_eval.py --save-baseline                  # record the baseline (C-212)
Results are appended to evals/results/history.jsonl (C-215) and written to --output.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import get_chat_client, get_retrieval_service  # noqa: E402
from app.core.config import RagMode, get_settings  # noqa: E402
from app.rag.evaluation.harness import (  # noqa: E402
    EvalRunner,
    append_history,
    check_thresholds,
    compare_baseline,
    load_golden,
)
from app.rag.graph.factory import get_graph_store  # noqa: E402
from app.rag.vectorstore.factory import get_vector_store  # noqa: E402

RESULTS = Path("evals/results")
COLUMNS = [
    "hit", "recall", "precision", "mrr", "ndcg", "citation_precision", "citation_recall",
    "faithfulness", "answer_relevance", "correctness", "routing_accuracy", "tool_selection",
    "latency_ms", "retrieval_ms", "tokens",
]  # fmt: skip


def print_table(report: dict) -> None:
    present = [c for c in COLUMNS if any(c in s for s in report["summary"].values())]
    width = max([10, *(len(label) + 2 for label in report["summary"])])
    print(f"{'mode':<{width}}" + "".join(f"{c[:12]:>13}" for c in present))
    for mode, summary in report["summary"].items():
        print(
            f"{mode:<{width}}"
            + "".join(f"{summary.get(c, float('nan')):>13.3f}" for c in present)
        )


async def evaluate(args, settings=None) -> dict:
    settings = settings or get_settings()
    service = get_retrieval_service(settings)
    cases = load_golden(args.golden)
    if args.rag_mode == "all":
        modes: list[RagMode | None] = list(service.registry.enabled_modes)
    elif isinstance(args.rag_mode, list):
        modes = [RagMode(m) for m in args.rag_mode]
    else:
        modes = [RagMode(args.rag_mode) if args.rag_mode else None]
    store_arg = getattr(args, "vector_store", None)
    if store_arg == "all":
        stores: list[str | None] = list(service.vector_backends) or [None]
    else:
        stores = [store_arg or None]
    runner = EvalRunner(
        service, llm=get_chat_client(settings) if args.judge else None, top_k=args.top_k
    )
    try:
        return await runner.run(cases, modes, judge=args.judge, vector_stores=stores)
    finally:
        if settings.vector_enabled:
            await get_vector_store(settings).close()
        if settings.graph_enabled:
            await get_graph_store(settings).close()


async def main(args) -> int:
    settings = get_settings()
    report = await evaluate(args, settings)
    print_table(report)
    for mode, by_type in report["by_type"].items():
        for question_type, summary in by_type.items():
            print(
                f"  [{mode}/{question_type}] hit={summary.get('hit', 0):.2f} "
                f"n={int(summary['cases'])}"
            )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    append_history(
        report,
        RESULTS / "history.jsonl",
        {"prompt_versions": settings.prompt_versions, "retrieval_mode": settings.retrieval_mode},
    )
    if args.save_baseline:
        (RESULTS / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("baseline saved")
    failures: list[str] = []
    if args.gate:
        failures += check_thresholds(report, args.thresholds)
        failures += [f"regression {r}" for r in compare_baseline(report, RESULTS / "baseline.json")]
        for failure in failures:
            print("GATE FAIL:", failure)
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RAG quality on a golden set")
    p.add_argument("--golden", default="evals/golden_set.jsonl")
    p.add_argument("--rag-mode", choices=[m.value for m in RagMode] + ["all"], default=None)
    p.add_argument(
        "--vector-store",
        default=None,
        help="mongo | all (every store in VECTOR_STORE_PROVIDERS); "
        "default: VECTOR_STORE_PROVIDER",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--judge", action="store_true", help="LLM-judge faithfulness/relevance/correctness"
    )
    p.add_argument(
        "--gate", action="store_true", help="Fail on threshold violations or regressions"
    )
    p.add_argument("--thresholds", default="evals/thresholds.yaml")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--output", default="evals/results/latest.json")
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parser().parse_args())))

"""Ablation runner (S-34..S-38, G-61, G-62, A-50): evaluate configuration variants side by
side - chunk size, embeddings, dense vs BM25 vs hybrid, rerankers, enrichment, graph
settings, agentic vs non-agentic - on the same golden set.

Variants are defined in evals/ablations.yaml. Variants that change indexing re-ingest into
their own collection first (cases are evaluated against that collection).

Usage:
    python scripts/run_ablation.py [--config evals/ablations.yaml] [--judge] [--only name]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval  # noqa: E402

from app.api import deps  # noqa: E402
from app.core.config import get_settings  # noqa: E402


async def run_variant(variant: dict, golden: str, judge: bool) -> dict:
    previous = {k: os.environ.get(k) for k in variant.get("env", {})}
    os.environ.update({k: str(v) for k, v in variant.get("env", {}).items()})
    get_settings.cache_clear()
    deps.reset()
    try:
        settings = get_settings()
        golden_path = golden
        if variant.get("ingest"):
            collection = variant.get("collection", f"ablation-{variant['name']}")
            pipeline = deps.build_ingestion_pipeline(settings)
            await pipeline.run(variant["ingest"]["source"], collection, force=True)
            # point every case at the variant's collection
            cases = [
                json.loads(line)
                for line in Path(golden).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            golden_path = f"evals/results/golden-{variant['name']}.jsonl"
            Path(golden_path).parent.mkdir(parents=True, exist_ok=True)
            Path(golden_path).write_text(
                "\n".join(json.dumps({**c, "collection": collection}) for c in cases),
                encoding="utf-8",
            )
        args = run_eval.parser().parse_args(
            ["--golden", golden_path, *(["--judge"] if judge else [])]
        )
        args.rag_mode = variant.get("modes") or None
        return await run_eval.evaluate(args, settings)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        deps.reset()


async def main(config_path: str, judge: bool, only: str | None) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows = []
    for variant in config["variants"]:
        if only and variant["name"] != only:
            continue
        report = await run_variant(variant, config.get("golden", "evals/golden_set.jsonl"), judge)
        for mode, summary in report["summary"].items():
            rows.append((variant["name"], mode, summary))
    metrics = [
        "hit",
        "recall",
        "mrr",
        "ndcg",
        "faithfulness",
        "correctness",
        "latency_ms",
        "tokens",
    ]
    print(f"{'variant':<24}{'mode':<10}" + "".join(f"{m:>12}" for m in metrics))
    for name, mode, summary in rows:
        print(
            f"{name:<24}{mode:<10}"
            + "".join(f"{summary.get(m, float('nan')):>12.3f}" for m in metrics)
        )
    Path("evals/results/ablation.json").write_text(
        json.dumps([{"variant": n, "mode": m, "summary": s} for n, m, s in rows], indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation variants")
    parser.add_argument("--config", default="evals/ablations.yaml")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--only")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.judge, args.only))

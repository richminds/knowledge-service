"""LLM-as-judge calibration (C-209): compare judge scores with human labels.

evals/human_labels.jsonl (labelled by reviewers on a 1-5 scale):
  {"question": ..., "answer": ..., "passages": [...], "groundedness": 4, "answer_relevance": 5}
Reports exact agreement, agreement within +-1 and Spearman rank correlation per metric.
Re-run whenever the judge prompt or model changes.

Usage:
    python scripts/calibrate_judge.py [--labels evals/human_labels.jsonl]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import get_chat_client  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.rag.evaluation.judge import AnswerJudge  # noqa: E402
from app.rag.vectorstore.base import VectorSearchResult  # noqa: E402


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ra, rb = _ranks(a), _ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var = (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return cov / var if var else 0.0


async def main(labels_path: str) -> None:
    judge = AnswerJudge(get_chat_client(get_settings()))
    labels = [
        json.loads(line)
        for line in Path(labels_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs: dict[str, list[tuple[float, float]]] = {"groundedness": [], "answer_relevance": []}
    for item in labels:
        passages = [
            VectorSearchResult(chunk_id=str(i), content=p, score=1.0, source="label")
            for i, p in enumerate(item["passages"])
        ]
        scores = await judge.evaluate(item["question"], item["answer"], passages)
        if scores is None:
            continue
        for metric in pairs:
            if metric in item:
                pairs[metric].append((float(item[metric]), float(getattr(scores, metric))))
    for metric, values in pairs.items():
        if not values:
            continue
        human, model = zip(*values, strict=True)
        exact = sum(h == m for h, m in values) / len(values)
        within = sum(abs(h - m) <= 1 for h, m in values) / len(values)
        print(
            f"{metric}: n={len(values)} exact={exact:.2f} within1={within:.2f} "
            f"spearman={spearman(list(human), list(model)):.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels")
    parser.add_argument("--labels", default="evals/human_labels.jsonl")
    asyncio.run(main(parser.parse_args().labels))

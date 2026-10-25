"""Phase 6: evaluation harness metrics, gate, baseline and an end-to-end run."""

import json
import sys
from pathlib import Path

import pytest

from app.core.config import RagMode
from app.rag.evaluation.harness import (
    EvalRunner,
    GoldenCase,
    append_history,
    check_thresholds,
    compare_baseline,
    load_golden,
    retrieval_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_retrieval_metrics():
    metrics = retrieval_metrics([False, True, True], total_relevant=2, k=3)
    assert metrics["recall"] == 1.0 and metrics["mrr"] == 0.5
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert 0 < metrics["ndcg"] < 1
    assert retrieval_metrics([False], 1, 3)["hit"] == 0.0


def test_gate_and_baseline(tmp_path):
    report = {"summary": {"semantic": {"hit": 0.6, "latency_ms": 2000.0, "cases": 3}}}
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("all: {hit: 0.7, latency_ms_max: 1000}\n")
    assert check_thresholds(report, thresholds) == [
        "semantic.hit=0.6 < 0.7",
        "semantic.latency_ms=2000.0 > 1000",
    ]
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"summary": {"semantic": {"hit": 0.8, "latency_ms": 1000.0}}}))
    assert compare_baseline(report, baseline) == [
        "semantic.hit: 0.8 -> 0.6",
        "semantic.latency_ms: 1000.0 -> 2000.0",
    ]
    append_history(report, tmp_path / "h.jsonl", {"note": "x"})
    assert json.loads((tmp_path / "h.jsonl").read_text())["summary"] == report["summary"]


def test_store_runs_use_their_mode_thresholds(tmp_path):
    report = {"summary": {"semantic@mongo": {"hit": 0.5, "cases": 1}}}
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("semantic: {hit: 0.7}\n")
    assert check_thresholds(report, thresholds) == ["semantic@mongo.hit=0.5 < 0.7"]


def test_load_golden_types(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"question": "Who runs X?", "question_type": "relationship", "extra": 1}\n')
    case = load_golden(path)[0]
    assert case.id == "case-1" and case.mode == "graph"


async def test_eval_runner_end_to_end(ingestion_pipeline, retrieval_service, chat_client):
    await ingestion_pipeline.run("docs", "default", metadata={"account_id": "acct-test"})
    cases = [
        GoldenCase(
            "How many annual leave days?",
            "a",
            question_type="factoid",
            expected_sources=["leave.md"],
            expected_answer="25 days",
        ),
        GoldenCase("What is the cafeteria menu?", "b", question_type="unanswerable"),
    ]
    report = await EvalRunner(retrieval_service, llm=chat_client, top_k=3).run(
        cases, [RagMode.SEMANTIC, RagMode.AUTO], judge=True
    )
    semantic = report["summary"]["semantic"]
    assert (
        semantic["hit"] == 1.0
        and semantic["correctness"] == 4.0
        and semantic["faithfulness"] == 5.0
    )
    assert "routing_accuracy" in report["summary"]["auto"]
    assert set(report["by_type"]["semantic"]) == {"factoid", "unanswerable"}


def test_spearman():
    from calibrate_judge import spearman

    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

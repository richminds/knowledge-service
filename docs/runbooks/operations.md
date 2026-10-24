# Operations & continuous improvement (C-237 – C-244)

| Cadence | Activity | How |
|---|---|---|
| Weekly | **Feedback triage** (C-237) | Review 👎 feedback (see the query below). Tag each item with the failing stage: `retrieval`, `ranking`, `generation`, `content_gap`, `access`. |
| Weekly | **Failures → regression tests** (C-238) | Add each confirmed failure to `evals/golden_set.jsonl`, with `expected_sources`. |
| Weekly | **Content gaps** (C-239) | Open the workbook's *Unanswered questions* panel and send the topics to content owners. |
| Monthly | **Retuning** (C-240) | `scripts/run_ablation.py` over top-k, retrieval mode, reranker and chunking. Change only what beats the baseline. |
| Monthly | **Cost review** (C-242) | Compare the budget with actual spend and tokens by purpose and mode. |
| Monthly | **Stakeholder report** (C-244) | Users, questions, the 👍 rate, the confidence mix and the answered rate (see the query below). |
| Per release of a model | **Model re-evaluation** (C-241) | Add a variant to `evals/ablations.yaml`, run it, and update the ADR if you switch. |
| Quarterly | **Judge calibration** (C-209) | Relabel about 50 answers in `evals/human_labels.jsonl` and run `scripts/calibrate_judge.py`. |
| Quarterly | **Red team** (C-201) | See [red-team.md](red-team.md). |

## Queries

Feedback with comments:

```kusto
traces | where message == "user_feedback" and tostring(customDimensions.rating) == "down"
| project timestamp, response_id = tostring(customDimensions.response_id),
          mode = tostring(customDimensions.rag_mode), comment = tostring(customDimensions.comment)
```

Monthly adoption and quality:

```kusto
customMetrics | where name == "rag.answers" and timestamp > ago(30d)
| summarize answers = sum(valueSum) by confidence = tostring(customDimensions.confidence),
            mode = tostring(customDimensions["rag.mode"])
```

More queries are in `infra/monitoring/rag-queries.kql`.

## Baseline management

- After an approved quality improvement, run `python scripts/run_eval.py --rag-mode all --judge --save-baseline` and commit `evals/results/baseline.json`. The CI gate compares against it.
- `evals/results/history.jsonl` keeps every run, with the prompt versions used (C-215).

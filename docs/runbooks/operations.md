# Operations & continuous improvement (C-237 – C-244)

| Cadence | Activity | How |
|---|---|---|
| Weekly | **Feedback triage** (C-237) | Review 👎 feedback (see the query below). Tag each item with the failing stage: `retrieval`, `ranking`, `generation`, `content_gap`, `access`. |
| Weekly | **Failures → regression tests** (C-238) | Add each confirmed failure to `evals/golden_set.jsonl`, with `expected_sources`. |
| Weekly | **Content gaps** (C-239) | Collect the unanswered questions from the `rag_answer` logs and send the topics to content owners. |
| Monthly | **Retuning** (C-240) | `scripts/run_ablation.py` over top-k, retrieval mode, reranker and chunking. Change only what beats the baseline. |
| Monthly | **Cost review** (C-242) | Compare the budget with actual spend and tokens by purpose in the LLM gateway's per-caller stats. |
| Monthly | **Stakeholder report** (C-244) | Users, questions, the 👍 rate, the confidence mix and the answered rate (see the query below). |
| When the gateway's models change | **Model re-evaluation** (C-241) | The LLM gateway picks the model, so re-run the golden set (`scripts/run_eval.py --rag-mode all --judge`) against the new configuration and compare it with the baseline. A new embedding model also needs a [re-index](reindex.md). |
| Quarterly | **Judge calibration** (C-209) | Relabel about 50 answers in `evals/human_labels.jsonl` and run `scripts/calibrate_judge.py`. |
| Quarterly | **Red team** (C-201) | See [red-team.md](red-team.md). |

## Log events

The service writes JSON logs to stdout; query them wherever your platform collects them.

- Feedback with comments: `user_feedback` events with `rating` = `down`, with their `response_id`, `rag_mode` and `comment`.
- Monthly adoption and quality: `rag_answer` events, counted by `confidence` and `rag_mode`.

## Baseline management

- After an approved quality improvement, run `python scripts/run_eval.py --rag-mode all --judge --save-baseline` and commit `evals/results/baseline.json`. `run_eval.py --gate` compares against it.
- `evals/results/history.jsonl` keeps every run, with the prompt versions used (C-215).

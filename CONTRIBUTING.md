# Contributing (C-027)

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env                              # then fill in the TODO values
make test
```

Tests never call Azure. They use the fakes in `tests/fakes.py`, and `tests/conftest.py` clears any Azure settings from your shell.

## Workflow

1. Branch from `develop` (`feature/<short-name>`).
2. Run `make check` (lint, type-check, tests) before pushing. CI runs the same checks.
3. Merging to `develop` deploys to QA and runs the quality gate. `main` deploys to prod after approval.
4. If you change prompts, retrieval or chunking, attach the `scripts/run_eval.py --rag-mode all --judge` output to the PR.

## Conventions

- Black and ruff, with a line length of 100. Follow the style of the surrounding code.
- Settings go in `src/app/core/config.py`, with a `TODO` placeholder and a comment. Secrets go in Key Vault, never in code or `.env.example` values.
- Every retriever and store method takes `RetrievalFilters` and must enforce `principals`. Add a test for each new one.
- Record significant technology choices as an ADR in `docs/adr/`.

## How to …

**Add a data source.**
1. Write a connector in `src/app/rag/ingestion/connectors/`. Subclass `Connector` and implement `iterate()`. Return the source ACLs and `last_modified`.
2. Register it in `connectors/__init__.py`.
3. Add its settings to `config.py`, and an entry to `config/sources.yaml`.
4. Add a test with a fake HTTP transport, following the existing connector tests.

**Add a file format.** Add a parser to `rag/ingestion/parsing.py`, map its type in `filetypes.py`, and add a fixture document test.

**Add an agent tool.** Add it to `rag/agents/tools.py` with a clear description, since tool choice depends on it. Pass the `filters` through, register it in `api/deps.py:build_agent_tools`, and add its name to `AGENTIC_TOOLS`.

**Change a prompt.** Add a new version to `src/app/prompts/<name>.yaml` (do not edit the existing one). Select it with `PROMPT_VERSIONS={"<name>": "v3"}`, evaluate it, then make it the default.

**Add a golden question.** Append it to `evals/golden_set.jsonl` with `question_type`, `expected_sources` and `expected_answer`.

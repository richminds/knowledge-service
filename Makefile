.PHONY: install dev run test lint format typecheck check docker-build docker-run ingest create-index eval eval-vector

install:
	pip install -r requirements-dev.txt
	pre-commit install

dev:
	uvicorn app.main:app --app-dir src --reload --port 8090

run:
	uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8090

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

typecheck:
	mypy src

check: lint typecheck test

docker-build:
	docker build -t oissa-rag-agent:local .

docker-run:
	docker compose up --build

ingest:
	python scripts/ingest_cli.py --source sample/ --collection default --targets vector,graph

create-index:
	python scripts/create_index.py

eval:
	python scripts/run_eval.py --golden evals/golden_set.jsonl --top-k 5 --rag-mode all

eval-vector:  ## every mode on every vector store (VECTOR_STORE_PROVIDERS), with the LLM judge
	python scripts/run_eval.py --golden evals/golden_set.jsonl --top-k 5 --rag-mode all --vector-store all --judge


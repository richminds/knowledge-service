.PHONY: install install-dev dev run test test-cov lint fmt docker docker-up docker-down token

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

# Reload loop on KNOWLEDGE_PORT (default 8090). Swagger: http://localhost:8090/docs
dev:
	$(PY) -m uvicorn app.main:app --reload --port $${KNOWLEDGE_PORT:-8090}

run:
	$(PY) run.py

test:
	$(PY) -m pytest -q

test-cov:
	$(PY) -m pytest --cov=app --cov=rag --cov=sdk --cov-report=term-missing

lint:
	$(PY) -m ruff check app rag sdk tests

fmt:
	$(PY) -m ruff check --fix app rag sdk tests

docker:
	docker build -t knowledge-service:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# Mint a JWT for local testing when RAG_AUTH_ENABLED=true.
token:
	$(PY) scripts/mint_token.py

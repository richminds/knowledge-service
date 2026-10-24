"""Load test (C-224): realistic chat traffic against a deployed environment.

Run against QA only (it calls the LLM gateway and costs tokens). Go through the API
gateway, which authenticates the token and injects the identity headers this service reads:

    pip install locust
    export RAG_BEARER_TOKEN=<auth-service access token>
    locust -f tests/load/locustfile.py --host https://<qa-api-gateway-route-to-this-service> \
           --users 20 --spawn-rate 2 --run-time 10m --headless --csv evals/results/load

Targets (TODO: confirm with C-007 non-functional requirements): p95 chat < 8 s,
p95 retrieval < 1.5 s, error rate < 1 %. Questions come from the golden set so answers are
realistic; the mix weights mirror expected production traffic.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_set.jsonl"
QUESTIONS = [
    json.loads(line)["question"]
    for line in GOLDEN.read_text(encoding="utf-8").splitlines()
    if line.strip()
] or ["What is the annual leave policy?"]
RAG_MODES = [m for m in os.getenv("RAG_LOAD_MODES", "").split(",") if m]  # e.g. semantic,graph


def _headers() -> dict[str, str]:
    if token := os.getenv("RAG_BEARER_TOKEN"):
        return {"Authorization": f"Bearer {token}"}
    return {}


class KnowledgeUser(HttpUser):
    wait_time = between(3, 10)  # think time between questions

    def on_start(self) -> None:
        self.client.headers.update(_headers())
        self.conversation_id: str | None = None

    def _body(self) -> dict:
        body: dict = {"query": random.choice(QUESTIONS)}  # noqa: S311 - not security sensitive
        if RAG_MODES:
            body["rag_mode"] = random.choice(RAG_MODES)  # noqa: S311
        return body

    @task(6)
    def chat_stream(self) -> None:
        body = {**self._body(), "conversation_id": self.conversation_id}
        with self.client.post(
            "/api/v1/chat/stream", json=body, stream=True, name="chat/stream", catch_response=True
        ) as response:
            text = "".join(chunk.decode("utf-8", "ignore") for chunk in response.iter_content(None))
            if response.status_code != 200 or "event: done" not in text:
                response.failure(f"status={response.status_code} incomplete stream")
            elif "event: error" in text:
                response.failure("stream reported an error")

    @task(3)
    def chat(self) -> None:
        with self.client.post(
            "/api/v1/chat", json=self._body(), name="chat", catch_response=True
        ) as r:
            if r.status_code == 200:
                self.conversation_id = r.json().get("conversation_id")
            elif r.status_code == 429:
                r.success()  # rate limiting is expected behaviour under load

    @task(1)
    def retrieval(self) -> None:
        self.client.post(
            "/api/v1/retrieval", json={"query": random.choice(QUESTIONS)}, name="retrieval"
        )  # noqa: S311


@events.quitting.add_listener
def _enforce_slo(environment, **_kwargs) -> None:
    """Non-zero exit when the error budget is exceeded (usable as a pipeline gate)."""
    stats = environment.stats.total
    if stats.num_requests and stats.fail_ratio > float(
        os.getenv("RAG_LOAD_MAX_FAIL_RATIO", "0.01")
    ):
        environment.process_exit_code = 1

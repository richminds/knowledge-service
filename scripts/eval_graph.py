"""Graph-specific evaluation (C7).

  extraction  entity / relation precision & recall on a labelled sample (G-57)
              evals/graph_extraction_gold.jsonl:
                {"text": ..., "entities": [names], "relations": [[source, TYPE, target]]}
  linking     entity-linking accuracy (G-58)
              evals/entity_linking_gold.jsonl: {"question": ..., "entities": [names]}
  cypher      template selection accuracy (G-59)
              evals/cypher_gold.jsonl: {"question": ..., "template": name or null}

Usage:
    python scripts/eval_graph.py extraction|linking|cypher|all [--collection default]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import build_graph_extractor, get_chat_client  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.rag.graph.base import entity_key  # noqa: E402
from app.rag.graph.cypher import CypherPlanner, load_templates  # noqa: E402
from app.rag.graph.factory import get_graph_store  # noqa: E402
from app.rag.types import RetrievalFilters  # noqa: E402


def load(path: str) -> list[dict]:
    file = Path(path)
    if not file.is_file():
        print(f"(skipped: {path} not found)")
        return []
    return [
        json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


async def eval_extraction(extractor) -> dict:
    ent_pred, ent_gold, rel_pred, rel_gold = set(), set(), set(), set()
    for i, case in enumerate(load("evals/graph_extraction_gold.jsonl")):
        entities, relations, _ = await extractor.extract_all(case["text"])
        ent_pred |= {(i, e.key) for e in entities}
        ent_gold |= {(i, entity_key(n)) for n in case["entities"]}
        rel_pred |= {(i, entity_key(r.source), r.type, entity_key(r.target)) for r in relations}
        rel_gold |= {(i, entity_key(s), t, entity_key(o)) for s, t, o in case["relations"]}
    return {"entities_prf": prf(ent_pred, ent_gold), "relations_prf": prf(rel_pred, rel_gold)}


async def eval_linking(extractor, store, collection: str) -> dict:
    cases = load("evals/entity_linking_gold.jsonl")
    correct = 0
    for case in cases:
        terms = await extractor.query_terms(case["question"])
        found = {
            e.key
            for e in await store.find_entities(terms, RetrievalFilters(collection=collection), 10)
        }
        correct += {entity_key(n) for n in case["entities"]} <= found
    return {"cases": len(cases), "accuracy": correct / len(cases) if cases else 0.0}


async def eval_cypher(planner) -> dict:
    cases = load("evals/cypher_gold.jsonl")
    correct = 0
    for case in cases:
        chosen = await planner.choose_template(case["question"])
        correct += (chosen[0].name if chosen else None) == case.get("template")
    return {"cases": len(cases), "template_accuracy": correct / len(cases) if cases else 0.0}


async def main(task: str, collection: str) -> None:
    settings = get_settings()
    extractor = build_graph_extractor(settings)
    if task in ("extraction", "all"):
        print("extraction:", await eval_extraction(extractor))
    if task in ("linking", "all"):
        store = get_graph_store(settings)
        try:
            print("linking:", await eval_linking(extractor, store, collection))
        finally:
            await store.close()
    if task in ("cypher", "all"):
        planner = CypherPlanner(
            get_chat_client(settings), load_templates(settings.graph_cypher_templates_path)
        )
        print("cypher:", await eval_cypher(planner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG evaluation")
    parser.add_argument("task", choices=["extraction", "linking", "cypher", "all"])
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()
    asyncio.run(main(args.task, args.collection))

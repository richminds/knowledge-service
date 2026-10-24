"""Synthetic question generation (C-204): expand golden-set coverage per question type.

Samples indexed documents (from the ingestion manifest) and asks the utility model to write
questions answerable from them:
  factoid       one chunk
  multi_hop     two chunks of the same document combined
  unanswerable  plausible questions the document does NOT answer
Output lines follow the golden-set schema with expected_sources set. Have subject-matter
experts review them before merging into evals/golden_set.jsonl (C-203).

Usage:
    python scripts/generate_questions.py --collection default --docs 20 --out evals/synthetic.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from app.api.deps import build_ingestion_pipeline, get_chat_client  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.rag.types import RetrievalFilters  # noqa: E402
from app.rag.vectorstore.factory import get_vector_store  # noqa: E402

PROMPT = (
    "Write evaluation questions for a document assistant from the passages below. Return JSON "
    '{{"factoid": [{{"question": str, "answer": str}}], "multi_hop": [{{"question": str, "answer": '
    "str}}], "
    '"unanswerable": [{{"question": str}}]}} with up to {n} of each. Questions must be natural, '
    "specific, and answerable only from the passages (except unanswerable ones)."
)


async def main(collection: str, docs: int, per_type: int, out: str) -> None:
    settings = get_settings()
    llm = get_chat_client(settings)
    store = get_vector_store(settings)
    manifest = build_ingestion_pipeline(settings).manifest
    entries = [
        e for e in await manifest.list_entries(collection) if e.status in ("indexed", "graphed")
    ]
    random.shuffle(entries)
    lines = []
    for entry in entries[:docs]:
        chunks = await store.get_document_chunks(
            entry.doc_id, RetrievalFilters(collection=collection), limit=20
        )
        if not chunks:
            continue
        sample = random.sample(chunks, min(2, len(chunks)))  # noqa: S311
        passages = "\n\n".join(c.content[:1500] for c in sample)
        data = await llm.complete_json(
            PROMPT.format(n=per_type), passages, purpose="synthetic_questions"
        )
        for qtype in ("factoid", "multi_hop", "unanswerable"):
            for item in data.get(qtype) or []:
                lines.append(
                    {
                        "question": item.get("question", ""),
                        "collection": collection,
                        "question_type": qtype,
                        "expected_sources": [] if qtype == "unanswerable" else [entry.source],
                        "expected_answer": item.get("answer", ""),
                        "synthetic": True,
                    }
                )
    Path(out).write_text(
        "\n".join(json.dumps(line) for line in lines if line["question"]), encoding="utf-8"
    )
    print(f"wrote {len(lines)} questions to {out} - review before adding to the golden set")
    await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation questions")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--docs", type=int, default=20)
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--out", default="evals/synthetic.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.docs, args.per_type, args.out))

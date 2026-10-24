"""In-memory stand-ins for Azure OpenAI, the MongoDB vector store and graph DB, for offline tests."""

import hashlib
import json
import re
from collections.abc import AsyncIterator, Collection
from dataclasses import asdict

from app.rag.generation.prompts import load_prompt
from app.rag.graph.base import ChunkGraph, Entity, GraphExport, GraphFact, GraphStore, entity_key
from app.rag.types import RetrievalFilters, is_allowed
from app.rag.vectorstore.base import (
    SEARCHABLE_KINDS,
    VectorRecord,
    VectorSearchResult,
    VectorStore,
)

_CAPITALISED = re.compile(r"\b[A-Z][a-z]{2,}\b")
_GREETING = re.compile(r"^\s*(hi|hello|thanks)\b", re.IGNORECASE)


class FakeEmbeddingClient:
    """Deterministic bag-of-words-ish embeddings: shared words => higher similarity."""

    dims = 64

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.strip(".,?!").encode()).hexdigest(), 16) % self.dims
            vector[bucket] += 1.0
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_text(self, text: str) -> list[float]:
        return self._embed(text)


def _matches(r: VectorRecord, filters: RetrievalFilters, kinds: tuple[str, ...] | None) -> bool:
    if r.collection != filters.collection or not filters.allows(r.acl):
        return False
    if kinds is not None and r.kind not in kinds:
        return False
    if filters.doc_types and r.doc_type not in filters.doc_types:
        return False
    if filters.languages and r.language not in filters.languages:
        return False
    if filters.sources and r.source not in filters.sources:
        return False
    if filters.tags and not set(r.tags) & filters.tags:
        return False
    if filters.doc_ids and r.doc_id not in filters.doc_ids:
        return False
    date = (r.effective_date or "")[:10]
    if filters.effective_from and (not date or date < filters.effective_from):
        return False
    return not (filters.effective_to and (not date or date > filters.effective_to))


def _result(r: VectorRecord, score: float) -> VectorSearchResult:
    data = asdict(r)
    for key in (
        "embedding",
        "collection",
        "content_hash",
        "doc_type",
        "language",
        "tags",
        "embedding_model",
        "ingest_version",
    ):
        data.pop(key)
    data["metadata"] = dict(r.metadata)
    return VectorSearchResult(score=score, **data)


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.search_calls: list[dict] = []

    async def upsert(self, records: list[VectorRecord]) -> int:
        for record in records:
            self.records[record.chunk_id] = record
        return len(records)

    async def search(
        self,
        query_embedding: list[float] | None,
        filters: RetrievalFilters,
        top_k: int,
        query_text: str | None = None,
        kinds: tuple[str, ...] = SEARCHABLE_KINDS,
        hybrid: bool | None = None,
    ) -> list[VectorSearchResult]:
        self.search_calls.append({"query_text": query_text, "hybrid": hybrid, "kinds": kinds})
        keyword_side = query_embedding is None or hybrid is not False
        words = set((query_text or "").lower().split()) if keyword_side else set()

        def score(r: VectorRecord) -> float:
            dense = (
                sum(x * y for x, y in zip(query_embedding, r.embedding or [], strict=False))
                if query_embedding is not None
                else 0.0
            )
            keyword = len(words & set(r.content.lower().split())) if words else 0
            return dense + keyword

        scored = [(score(r), r) for r in self.records.values() if _matches(r, filters, kinds)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_result(r, s) for s, r in scored[:top_k] if s > 0]

    async def get_by_ids(
        self, ids: list[str], filters: RetrievalFilters
    ) -> list[VectorSearchResult]:
        return [
            _result(self.records[i], 0.0)
            for i in ids
            if i in self.records and _matches(self.records[i], filters, None)
        ]

    async def get_document_chunks(
        self, doc_id: str, filters: RetrievalFilters, limit: int = 200
    ) -> list[VectorSearchResult]:
        chunks = [
            r
            for r in self.records.values()
            if r.doc_id == doc_id and _matches(r, filters, ("chunk",))
        ]
        return [_result(r, 0.0) for r in sorted(chunks, key=lambda r: r.chunk_index)][:limit]

    async def count(self, collection: str, source: str | None = None) -> int:
        return sum(
            1
            for r in self.records.values()
            if r.collection == collection and (source is None or r.source == source)
        )

    async def update_acl(self, source: str, collection: str, acl: list[str]) -> int:
        updated = 0
        for r in self.records.values():
            if r.source == source and r.collection == collection:
                r.acl = list(acl)
                updated += 1
        return updated

    async def delete_by_source(
        self,
        source: str,
        collection: str,
        except_version: str | None = None,
        keep_ids: Collection[str] = (),
    ) -> int:
        doomed = [
            k
            for k, r in self.records.items()
            if r.source == source
            and r.collection == collection
            and (except_version is None or r.ingest_version != except_version)
            and k not in keep_ids
        ]
        for k in doomed:
            del self.records[k]
        return len(doomed)

    async def delete_collection(self, collection: str) -> int:
        doomed = [k for k, r in self.records.items() if r.collection == collection]
        for k in doomed:
            del self.records[k]
        return len(doomed)


class FakeGraphStore(GraphStore):
    """In-memory graph with the same ACL semantics as the MongoDB graph store."""

    supports_cypher = True  # lets the tests exercise the strategy's Cypher path

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkGraph] = {}
        self.embeddings: dict[tuple[str, str], list[float]] = {}
        self.properties: dict[tuple[str, str], dict] = {}
        self.cypher_rows: list[tuple] = []  # what run_cypher returns in tests

    async def upsert_chunk_graph(self, chunk: ChunkGraph) -> None:
        self.chunks[chunk.chunk_id] = chunk

    def _visible(self, filters: RetrievalFilters) -> list[ChunkGraph]:
        return [
            c
            for c in self.chunks.values()
            if c.collection == filters.collection and filters.allows(c.acl)
        ]

    def _entity(self, entity: Entity, collection: str) -> Entity:
        props = self.properties.get((collection, entity.key), {})
        return Entity(
            entity.name,
            entity.type,
            entity.description,
            list(entity.aliases),
            entity.external_id,
            rank=float(props.get("rank", 0.0)),
            community=props.get("community_0"),
        )

    async def find_entities(
        self, terms: list[str], filters: RetrievalFilters, limit: int
    ) -> list[Entity]:
        found: dict[str, Entity] = {}
        for chunk in self._visible(filters):
            for entity in chunk.entities:
                aliases = {entity_key(a) for a in entity.aliases}
                for term in terms:
                    if term and (term in entity.key or term in aliases):
                        match = self._entity(entity, filters.collection)
                        match.score = 1.0 if term in (entity.key, *aliases) else 0.6
                        if entity.key not in found or match.score > found[entity.key].score:
                            found[entity.key] = match
        return sorted(found.values(), key=lambda e: -e.score)[:limit]

    async def neighborhood(
        self,
        entity_keys: list[str],
        filters: RetrievalFilters,
        hops: int,
        limit: int,
        relation_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[GraphFact]:
        facts: dict[tuple, GraphFact] = {}
        frontier, seen = set(entity_keys), set(entity_keys)
        for _ in range(hops):
            nxt: set[str] = set()
            for chunk in self._visible(filters):
                for r in chunk.relations:
                    src, dst = entity_key(r.source), entity_key(r.target)
                    if relation_types and r.type not in relation_types:
                        continue
                    if r.weight < min_weight or not (src in frontier or dst in frontier):
                        continue
                    facts.setdefault(
                        (src, r.type, dst),
                        GraphFact(
                            r.source,
                            r.type,
                            r.target,
                            r.description,
                            src,
                            dst,
                            chunk.chunk_id,
                            r.weight,
                            list(chunk.acl),
                        ),
                    )
                    nxt |= {src, dst} - seen
            seen |= nxt
            frontier = nxt
        return list(facts.values())[:limit]

    async def chunks_for_entities(
        self, entity_keys: list[str], filters: RetrievalFilters, limit: int
    ) -> list[VectorSearchResult]:
        scored = []
        for chunk in self._visible(filters):
            hits = len({e.key for e in chunk.entities} & set(entity_keys))
            if hits:
                scored.append((hits, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            VectorSearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=hits / len(entity_keys),
                source=c.source,
                metadata={"retriever": "graph"},
                acl=list(c.acl),
                doc_id=c.doc_id,
                title=c.title,
            )
            for hits, c in scored[:limit]
        ]

    async def entities_for_chunks(self, chunk_ids, filters):
        return [
            self._entity(e, filters.collection)
            for c in self._visible(filters)
            if c.chunk_id in chunk_ids
            for e in c.entities
        ]

    async def claims_for_entities(self, entity_keys, filters, limit):
        return [
            (claim, c.chunk_id)
            for c in self._visible(filters)
            for claim in c.claims
            if entity_key(claim.subject) in entity_keys
        ][:limit]

    async def upsert_entity_embeddings(self, collection, items):
        for key, _, embedding in items:
            self.embeddings[(collection, key)] = embedding

    async def similar_entities(self, embedding, filters, limit):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
            return dot / norm if norm else 0.0

        visible = {e.key: e for c in self._visible(filters) for e in c.entities}
        scored = []
        for (collection, key), vector in self.embeddings.items():
            if collection == filters.collection and key in visible:
                entity = self._entity(visible[key], collection)
                entity.score = cosine(embedding, vector)
                scored.append(entity)
        return sorted(scored, key=lambda e: -e.score)[:limit]

    async def export_graph(self, collection, acl_scope):
        export = GraphExport()
        for chunk in self.chunks.values():
            if chunk.collection != collection or not is_allowed(chunk.acl, acl_scope):
                continue
            for e in chunk.entities:
                export.nodes.setdefault(e.key, self._entity(e, collection))
            for r in chunk.relations:
                src, dst = entity_key(r.source), entity_key(r.target)
                export.edges.append(
                    GraphFact(
                        r.source,
                        r.type,
                        r.target,
                        r.description,
                        src,
                        dst,
                        chunk.chunk_id,
                        r.weight,
                        list(chunk.acl),
                    )
                )
        return export

    async def set_entity_properties(self, collection, properties):
        for key, values in properties.items():
            self.properties.setdefault((collection, key), {}).update(values)

    async def merge_entities(self, collection, keep, merge):
        keep_name = None
        for chunk in self.chunks.values():
            for e in chunk.entities:
                if e.key == keep:
                    keep_name = e.name
        merged = 0
        for chunk in self.chunks.values():
            if chunk.collection != collection:
                continue
            for e in chunk.entities:
                if e.key in merge and keep_name:
                    e.aliases = sorted({*e.aliases, e.name})
                    e.name = keep_name
                    merged += 1
            for r in chunk.relations:
                if entity_key(r.source) in merge:
                    r.source = keep_name
                if entity_key(r.target) in merge:
                    r.target = keep_name
        return len(merge) if merged else 0

    async def update_acl(self, source, collection, acl):
        n = 0
        for c in self.chunks.values():
            if c.source == source and c.collection == collection:
                c.acl = list(acl)
                n += 1
        return n

    async def run_cypher(self, query, params, columns):
        return list(self.cypher_rows)

    async def delete_by_source(self, source: str, collection: str) -> int:
        doomed = [
            k for k, c in self.chunks.items() if c.source == source and c.collection == collection
        ]
        for k in doomed:
            del self.chunks[k]
        return len(doomed)

    async def delete_collection(self, collection: str) -> int:
        doomed = [k for k, c in self.chunks.items() if c.collection == collection]
        for k in doomed:
            del self.chunks[k]
        return len(doomed)


class FakeChatClient:
    """Answers generation calls and the structured (JSON) helper calls deterministically.

    Graph extraction treats Capitalised words as entities linked in sequence; the agent
    routes to every tool, keeps every passage, and appends "details" when rewriting.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.json_calls: list[str] = []
        self.grade_keep: set[int] | None = None  # None = keep everything
        self.answer_template = "Answer to '{query}' [1]"
        self.analysis: dict | None = None  # override query_analysis output
        self.cypher_choice: dict | None = None
        self.agent_check: dict = {
            "grounded": True,
            "addresses_question": True,
            "unsupported": [],
            "missing": "",
        }
        self.agent_plan: list[str] = []
        self.route: dict | None = None
        self.cypher_generated: dict | None = None
        self.judge_scores: dict = {
            "groundedness": 5,
            "answer_relevance": 4,
            "context_relevance": 4,
            "reasoning": "Supported by passage 1.",
            "unsupported_claims": [],
        }

    def prompt(self, name: str):
        return load_prompt(name)

    async def generate_answer(self, query, contexts, history=(), prompt="rag_answer") -> str:
        self.calls.append({"query": query, "contexts": contexts, "history": list(history)})
        return self.answer_template.format(query=query)

    async def stream_answer(
        self, query, contexts, history=(), prompt="rag_answer"
    ) -> AsyncIterator[str]:
        for token in ["Answer ", "[1]"]:
            yield token

    async def complete_text(self, prompt: str, *, purpose: str, **values) -> str:
        self.json_calls.append(purpose)
        return "Hello! How can I help?"

    async def complete_json(self, system: str, user: str, *, purpose: str, **_) -> dict:
        self.json_calls.append(purpose)
        if purpose == "graph_extraction":
            names = list(dict.fromkeys(_CAPITALISED.findall(user.removeprefix("Text:\n"))))
            return {
                "entities": [{"name": n, "type": "Concept", "description": ""} for n in names],
                "relations": [
                    {"source": a, "target": b, "type": "RELATED_TO", "description": ""}
                    for a, b in zip(names, names[1:], strict=False)
                ],
            }
        if purpose == "query_analysis":
            if self.analysis is not None:
                return dict(self.analysis)
            question = user.rsplit("Latest question:\n", 1)[-1]
            return {
                "standalone_question": question,
                "intent": "chitchat" if _GREETING.match(question) else "rag",
                "complexity": "simple",
                "filters": {},
            }
        if purpose == "query_entities":
            themes = ["governance"] if "themes" in user.lower() or "overall" in user.lower() else []
            return {"entities": _CAPITALISED.findall(user), "themes": themes}
        if purpose == "graph_gleaning":
            return {"entities": [], "relations": []}
        if purpose == "community_report":
            names = re.findall(r"^- ([^(]+) \(", user, flags=re.M)
            return {
                "title": " & ".join(n.strip() for n in names[:2]),
                "summary": "Community about " + ", ".join(n.strip() for n in names),
                "rating": 7,
            }
        if purpose == "global_map":
            report = user.split("Report:\n", 1)[-1]
            return {"points": [{"text": report.splitlines()[0][:80], "score": 80}]}
        if purpose == "cypher_template":
            return dict(self.cypher_choice or {"template": None})
        if purpose == "cypher_generate":
            return dict(self.cypher_generated or {"cypher": "", "columns": []})
        if purpose in ("query_multi", "query_step_back", "query_hyde", "query_decompose"):
            return {
                "queries": [user + " overview"],
                "question": user + " in general",
                "passage": "Policy passage about " + user,
                "sub_questions": [user],
            }
        if purpose == "doc_profile":
            words = re.findall(r"[A-Za-z]{4,}", user)
            return {
                "summary": " ".join(words[:12]) + ".",
                "doc_type": "policy",
                "effective_date": "2024-07-01",
                "tags": ["leave"],
            }
        if purpose == "chunk_context":
            words = re.findall(r"[A-Za-z]{4,}", user)
            return {"context": f"This passage covers {' '.join(words[:3])}."}
        if purpose == "chunk_questions":
            words = re.findall(r"[A-Za-z]{4,}", user)
            return {
                "questions": [f"What about {w}?" for w in words[:2]] + ["Unrelated zebra question?"]
            }
        if purpose == "chunk_summary":
            return {"summary": user[:60]}
        if purpose == "agent_route":
            return dict(self.route or {"tools": ["vector", "graph"], "reason": "test"})
        if purpose == "agent_grade":
            numbers = {int(n) for n in re.findall(r"^\[(\d+)\]", user, flags=re.M)}
            keep = numbers if self.grade_keep is None else numbers & self.grade_keep
            return {"relevant": sorted(keep)}
        if purpose == "quality_judge":
            return dict(self.judge_scores)
        if purpose == "rerank":
            numbers = re.findall(r"^\[(\d+)\]", user, flags=re.M)
            return {"scores": {n: 10 - int(n) for n in numbers}}
        if purpose == "correctness_judge":
            return {"correctness": 4, "reason": "close"}
        if purpose == "agent_check":
            return dict(self.agent_check)
        if purpose == "agent_hop":
            return {"query": None}
        if purpose == "agent_plan":
            return {"sub_questions": self.agent_plan}
        if purpose == "agent_rewrite":
            return {"query": json.loads(user)["question"] + " details"}
        raise AssertionError(f"unexpected purpose {purpose}")

"""Structured graph questions: Cypher templates (G-45) and text-to-Cypher (G-43) with
validation and safety (G-44).

Templates (config/cypher_templates.yaml) are parameterised read-only queries for frequent
question patterns. Every template returns the standard fact columns including the
relation's ACL, chunk id and app account, so results are ACL- and account-filtered and
citable.

Free-form text-to-Cypher cannot be ACL-filtered generically (arbitrary projections), so it
only runs when ACLs are not enforced or the caller holds GRAPH_CYPHER_ROLE; otherwise the
template path is used. Generated queries are validated: read-only clauses only, the
collection parameter must be used, a LIMIT is enforced, and execution has a timeout.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.exceptions import GenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)

FACT_COLUMNS = (
    "source_name",
    "relation",
    "target_name",
    "description",
    "acl",
    "chunk_id",
    "account_id",
)
_WRITE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL|LOAD|FOREACH|UNWIND\s+.*\s+CREATE)\b",
    re.IGNORECASE,
)
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"")

TEMPLATE_PROMPT = (
    "Choose the query template that answers the question and fill its parameters. "
    "Templates:\n{templates}\n"
    'Return JSON: {{"template": name or null, "params": {{...}}}}. Use null when none fits. '
    "Parameter values must be entity names as written in the question."
)
CYPHER_PROMPT = (
    "Write ONE read-only openCypher query (openCypher dialect) answering the question over this "
    "graph schema:\n{schema}\n"
    "Rules: only MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT; every node "
    "pattern must include {{collection: $collection}}; entity keys are lower-case names; "
    'return at most 6 columns; end with LIMIT. Return JSON: {{"cypher": str, "columns": [names]}}.'
)
SCHEMA = (
    "(:Entity {key, name, type, description, rank})-[:RELATED {type, description, weight, acl, "
    "chunk_id, account_id}]->(:Entity)\n"
    "(:Chunk {chunk_id, content, source, title, acl})-[:MENTIONS]->(:Entity)\n"
    "(:Claim {text, date, acl, chunk_id})-[:ABOUT]->(:Entity)"
)


class CypherValidationError(ValueError):
    pass


def validate_cypher(query: str, max_rows: int = 50) -> str:
    """G-44: reject writes and procedure calls, require the collection scope, cap rows."""
    stripped = query.strip().rstrip(";")
    if ";" in stripped or "$$" in stripped or "$cypher$" in stripped.lower():
        raise CypherValidationError("Multiple statements or dollar quoting are not allowed")
    code = _STRING_LITERAL.sub("''", stripped)  # ignore keywords inside string literals
    if _WRITE.search(code):
        raise CypherValidationError("Only read-only queries are allowed")
    if not re.match(r"^\s*(OPTIONAL\s+)?MATCH\b", code, re.IGNORECASE):
        raise CypherValidationError("Query must start with MATCH")
    if "$collection" not in code:
        raise CypherValidationError("Query must be scoped with $collection")
    if not re.search(r"\bRETURN\b", code, re.IGNORECASE):
        raise CypherValidationError("Query must RETURN results")
    match = _LIMIT.search(code)
    if match:
        if int(match.group(1)) > max_rows:
            stripped = _LIMIT.sub(f"LIMIT {max_rows}", stripped)
    else:
        stripped = f"{stripped} LIMIT {max_rows}"
    return stripped


@dataclass
class CypherTemplate:
    name: str
    description: str
    query: str
    params: list[str]


@lru_cache(maxsize=4)
def load_templates(path: str) -> tuple[CypherTemplate, ...]:
    file = Path(path)
    if not file.is_file():
        return ()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return tuple(
        CypherTemplate(
            name,
            spec["description"],
            validate_cypher(spec["query"]),
            list(spec.get("params") or []),
        )
        for name, spec in (data.get("templates") or {}).items()
    )


class CypherPlanner:
    def __init__(self, llm, templates: tuple[CypherTemplate, ...]) -> None:
        self._llm = llm
        self._templates = {t.name: t for t in templates}

    async def choose_template(self, question: str) -> tuple[CypherTemplate, dict] | None:
        if not self._templates:
            return None
        listing = "\n".join(
            f"- {t.name}({', '.join(t.params)}): {t.description}" for t in self._templates.values()
        )
        try:
            data = await self._llm.complete_json(
                TEMPLATE_PROMPT.format(templates=listing), question, purpose="cypher_template"
            )
        except GenerationError:
            return None
        template = self._templates.get(str(data.get("template") or ""))
        if template is None:
            return None
        params = {
            p: str((data.get("params") or {}).get(p, "")).strip().lower() for p in template.params
        }
        if not all(params.values()):
            return None
        return template, params

    async def generate(self, question: str) -> tuple[str, tuple[str, ...]] | None:
        """G-43 free-form text-to-Cypher; returns a validated query and its column names."""
        try:
            data = await self._llm.complete_json(
                CYPHER_PROMPT.format(schema=SCHEMA), question, purpose="cypher_generate"
            )
            query = validate_cypher(str(data.get("cypher") or ""))
        except (GenerationError, CypherValidationError) as exc:
            logger.info("cypher_generation_rejected", error=str(exc))
            return None
        columns = tuple(
            re.sub(r"\W", "_", str(c))[:30] or f"c{i}"
            for i, c in enumerate(data.get("columns") or [])
        )
        returned = re.search(r"\bRETURN\b(.*?)(\bORDER\b|\bLIMIT\b|$)", query, re.IGNORECASE | re.S)
        expected = len([p for p in (returned.group(1) if returned else "").split(",") if p.strip()])
        if len(columns) != expected:
            columns = tuple(f"c{i}" for i in range(expected))
        return query, columns

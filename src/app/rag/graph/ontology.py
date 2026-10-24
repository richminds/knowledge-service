"""Graph ontology (G-03, G-04, G-09, G-14) and entity resolution (G-17, G-18).

The ontology (config/ontology.yaml) defines allowed entity and relation types, their
synonyms and few-shot examples. Extraction output is validated against it: synonyms are
remapped, unknown types become Concept / RELATED_TO, or are dropped when strict.

Resolution (config/entity_aliases.yaml): aliases map to one canonical entity (and an
external master-data id); names are normalised (case, punctuation, legal suffixes) so
"ACME Inc." and "Acme" resolve to the same node.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

FALLBACK_ENTITY = "Concept"
FALLBACK_RELATION = "RELATED_TO"
_LEGAL_SUFFIX = re.compile(
    r"[,.]?\s+(inc|ltd|llc|plc|corp|corporation|co|gmbh|sa|ag)\.?$", re.IGNORECASE
)


@dataclass
class Ontology:
    entity_types: dict[str, str] = field(default_factory=dict)  # name -> description
    relation_types: dict[str, str] = field(default_factory=dict)
    entity_synonyms: dict[str, str] = field(default_factory=dict)  # lower synonym -> type
    relation_synonyms: dict[str, str] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    strict: bool = False
    version: str = "0"

    def entity_type(self, raw: str) -> str | None:
        if not self.entity_types:
            return raw or FALLBACK_ENTITY
        cleaned = re.sub(r"[^A-Za-z]", "", raw or "")
        for name in self.entity_types:
            if name.lower() == cleaned.lower():
                return name
        mapped = self.entity_synonyms.get(cleaned.lower())
        if mapped:
            return mapped
        return None if self.strict else FALLBACK_ENTITY

    def relation_type(self, raw: str) -> str | None:
        cleaned = re.sub(r"[^A-Z0-9_]", "_", (raw or "").upper()).strip("_")
        if not self.relation_types:
            return cleaned or FALLBACK_RELATION
        if cleaned in self.relation_types:
            return cleaned
        mapped = self.relation_synonyms.get(cleaned)
        if mapped:
            return mapped
        return None if self.strict else FALLBACK_RELATION

    def prompt_section(self) -> str:
        if not self.entity_types:
            return ""
        entities = "\n".join(f"- {n}: {d}" for n, d in self.entity_types.items())
        relations = "\n".join(f"- {n}: {d}" for n, d in self.relation_types.items())
        return f"Entity types:\n{entities}\nRelation types:\n{relations}\n"


@lru_cache(maxsize=4)
def load_ontology(path: str) -> Ontology:
    file = Path(path)
    if not file.is_file():
        return Ontology()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    ontology = Ontology(strict=bool(data.get("strict")), version=str(data.get("version", "1")))
    for name, spec in (data.get("entity_types") or {}).items():
        ontology.entity_types[name] = (spec or {}).get("description", "")
        for synonym in (spec or {}).get("synonyms") or []:
            ontology.entity_synonyms[re.sub(r"[^A-Za-z]", "", synonym).lower()] = name
    for name, spec in (data.get("relation_types") or {}).items():
        ontology.relation_types[name] = (spec or {}).get("description", "")
        for synonym in (spec or {}).get("synonyms") or []:
            ontology.relation_synonyms[synonym.upper()] = name
    ontology.examples = list(data.get("examples") or [])
    return ontology


@dataclass
class CanonicalEntity:
    name: str
    type: str = ""
    external_id: str = ""


class EntityResolver:
    """Maps surface names to canonical entities (G-17, G-18)."""

    def __init__(self, aliases: dict[str, CanonicalEntity] | None = None) -> None:
        self._aliases = aliases or {}

    @staticmethod
    def normalise(name: str) -> str:
        text = re.sub(r"\s+", " ", name).strip()
        text = _LEGAL_SUFFIX.sub("", text)
        return re.sub(r"^the\s+", "", text, flags=re.IGNORECASE).strip(" .,;:'\"")

    def resolve(self, name: str) -> CanonicalEntity:
        normal = self.normalise(name)
        return self._aliases.get(normal.lower()) or CanonicalEntity(name=normal)


@lru_cache(maxsize=4)
def load_resolver(path: str) -> EntityResolver:
    file = Path(path)
    if not file.is_file():
        return EntityResolver()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    aliases: dict[str, CanonicalEntity] = {}
    for entry in data.get("entities") or []:
        canonical = CanonicalEntity(entry["name"], entry.get("type", ""), str(entry.get("id", "")))
        for alias in [entry["name"], *(entry.get("aliases") or [])]:
            aliases[EntityResolver.normalise(str(alias)).lower()] = canonical
    return EntityResolver(aliases)

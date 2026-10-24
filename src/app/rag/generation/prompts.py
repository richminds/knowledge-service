"""Prompt registry (C-179) and context formatting (C-175, C-176).

Prompts live in src/app/prompts/<name>.yaml with numbered versions; PROMPT_VERSIONS picks
the version per prompt (default: the file's `default`). Responses report the version used
so quality can be compared per version (A/B) and a bad version rolled back by config.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.rag.vectorstore.base import VectorSearchResult

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

NO_CONTEXT_ANSWER = (
    "I couldn't find information about this in the knowledge base. Try rephrasing the "
    "question with more specific terms (a policy or document name, a date, a department), "
    "or check that you have access to the relevant documents."
)
BLOCKED_INPUT_ANSWER = "I can't help with that request."
BLOCKED_OUTPUT_ANSWER = (
    "I generated an answer but withheld it because it may contain sensitive or unsafe "
    "content. Please rephrase the question or contact the document owner."
)
OUT_OF_SCOPE_ANSWER = (
    "That question is outside what I can help with. I answer questions about the "
    "organisation's documents and knowledge base."
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    def render(self, **values: str) -> tuple[str, str]:
        return self.system.format_map(_Defaults(values)), self.user.format_map(_Defaults(values))


class _Defaults(dict):
    """format_map helper: unknown placeholders render empty instead of raising."""

    def __missing__(self, key: str) -> str:
        return ""


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str | None = None) -> PromptTemplate:
    data = yaml.safe_load((PROMPT_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    chosen = version or data["default"]
    if chosen not in data["versions"]:
        raise ValueError(f"Prompt {name!r} has no version {chosen!r}")
    entry = data["versions"][chosen]
    return PromptTemplate(name, chosen, entry["system"], entry["user"])


def format_context(contexts: list[VectorSearchResult]) -> str:
    """Number each passage so the model can cite it as [n]; n maps to citations[n-1].
    Each block carries title/section/date/page so the model can judge recency and
    authority, and is fenced as data (<passage>) so its text is not taken as instructions."""
    blocks = []
    for i, chunk in enumerate(contexts, start=1):
        header = [f"source: {chunk.title or chunk.source}"]
        if chunk.section_path:
            header.append(f"section: {chunk.section_path}")
        if chunk.effective_date:
            header.append(f"date: {chunk.effective_date[:10]}")
        if chunk.page:
            header.append(f"page: {chunk.page}")
        blocks.append(f"[{i}] ({'; '.join(header)})\n<passage>\n{chunk.content}\n</passage>")
    return "\n\n".join(blocks)


def build_context_prompt(query: str, contexts: list[VectorSearchResult]) -> str:
    """User message for the default answer prompt (kept for callers without a template)."""
    return load_prompt("rag_answer").render(context=format_context(contexts), question=query)[1]

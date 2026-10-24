"""Structure-aware chunking (A8, S-01).

- chunks never span two sections (C-111); the section breadcrumb travels with each chunk
- prose is packed paragraph-by-paragraph and split at sentence, then word boundaries
  (C-110, C-115), counted with the model tokenizer (C-113)
- tables (header row repeated), code (whole definitions), slides (one per chunk) and
  transcripts get dedicated chunkers (C-112)
- overlap carries the trailing sentences of a chunk into the next one (C-114)
- tiny fragments merge into their neighbour (C-116)
- deterministic ids (C-117), prev/next links (C-118), pages and offsets (C-120),
  "Title > Section" prefix for embedding/BM25 (C-121)
- optional parent sections for small-to-big retrieval (S-01)
"""

import re
from dataclasses import dataclass

import tiktoken

from app.rag.ingestion.models import Chunk, Element, ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(À-ɏЀ-ӿ])|(?<=[。！？])")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class ChunkingOptions:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_tokens: int = 40
    parent_size: int = 0  # > 0 enables parent-child chunking (S-01), e.g. 1500
    prepend_header: bool = True
    strategy: str = "structure"  # structure | semantic (S-02: topic-shift boundaries)
    semantic_percentile: float = 0.9  # break where similarity drop is in the top 10%


# -- legacy helper (token sliding window) ------------------------------------------------------
@dataclass
class TextChunk:
    text: str
    index: int
    token_count: int


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[TextChunk]:
    """Token sliding window, kept for plain-text callers."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    tokens = _ENCODING.encode(text)
    chunks, start, index = [], 0, 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(TextChunk(_ENCODING.decode(tokens[start:end]), index, end - start))
        index += 1
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap
    return chunks


# -- splitting -------------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def split_to_fit(text: str, max_tokens: int) -> list[str]:
    """Recursive split: paragraph -> sentences -> words (C-110, C-115)."""
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces = split_sentences(text)
    if len(pieces) == 1:  # one huge sentence: fall back to words
        words, out, current = text.split(), [], []
        for word in words:
            if current and count_tokens(" ".join([*current, word])) > max_tokens:
                out.append(" ".join(current))
                current = []
            current.append(word)
        return out + ([" ".join(current)] if current else [])
    out, current = [], ""
    for sentence in pieces:
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            out.extend(split_to_fit(current, max_tokens))
            current = sentence
        else:
            current = candidate
    if current:
        out.extend(split_to_fit(current, max_tokens))
    return out


def _tail(text: str, max_tokens: int) -> str:
    """Trailing whole sentences up to max_tokens, used as overlap (C-114)."""
    if max_tokens <= 0:
        return ""
    tail: list[str] = []
    for sentence in reversed(split_sentences(text)):
        if count_tokens(" ".join([sentence, *tail])) > max_tokens:
            break
        tail.insert(0, sentence)
    return " ".join(tail)


def _split_table(text: str, max_tokens: int) -> list[str]:
    """Markdown tables split by rows with the header repeated (C-112)."""
    lines = text.splitlines()
    if count_tokens(text) <= max_tokens or len(lines) < 3:
        return split_to_fit(text, max_tokens) if count_tokens(text) > max_tokens else [text]
    header = lines[:2] if set(lines[1].replace("|", "").strip()) <= {"-", " ", ":"} else lines[:1]
    out, rows = [], []
    for line in lines[len(header) :]:
        if rows and count_tokens("\n".join([*header, *rows, line])) > max_tokens:
            out.append("\n".join([*header, *rows]))
            rows = []
        rows.append(line)
    if rows:
        out.append("\n".join([*header, *rows]))
    return out


# -- semantic chunking (S-02) -----------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def semantic_breakpoints(embeddings: list[list[float]], percentile: float = 0.9) -> list[int]:
    """Indices i where a new segment starts before sentence i: the drop in similarity
    between sentence i-1 and i is in the top (1 - percentile) of drops in the text."""
    if len(embeddings) < 3:
        return []
    distances = [1 - _cosine(a, b) for a, b in zip(embeddings, embeddings[1:], strict=False)]
    threshold = sorted(distances)[min(len(distances) - 1, int(len(distances) * percentile))]
    return [i + 1 for i, d in enumerate(distances) if d >= threshold and d > 0]


async def apply_semantic_boundaries(
    parsed: ParsedDocument, embedder, percentile: float = 0.9
) -> None:
    """Split long prose elements at topic shifts, marking new segments as boundaries."""
    elements: list[Element] = []
    for element in parsed.elements:
        sentences = split_sentences(element.text) if element.type == "paragraph" else []
        if len(sentences) < 4:
            elements.append(element)
            continue
        breaks = set(semantic_breakpoints(await embedder.embed_batch(sentences), percentile))
        segment: list[str] = []
        first = True
        for i, sentence in enumerate(sentences):
            if i in breaks and segment:
                elements.append(
                    Element(
                        "paragraph",
                        " ".join(segment),
                        page=element.page,
                        section_path=element.section_path,
                        boundary=not first,
                    )
                )
                segment, first = [], False
            segment.append(sentence)
        elements.append(
            Element(
                "paragraph",
                " ".join(segment),
                page=element.page,
                section_path=element.section_path,
                boundary=not first,
            )
        )
    parsed.elements = elements


# -- chunker ---------------------------------------------------------------------------------
@dataclass
class _Piece:
    text: str
    elements: list[Element]
    section: list[str]
    kind: str  # prose | table | code | slide | transcript


def _sections(elements: list[Element]) -> list[list[Element]]:
    groups: list[list[Element]] = []
    for element in elements:
        if element.type in ("title", "heading"):
            groups.append([element])
        elif groups and groups[-1][0].section_path == element.section_path:
            groups[-1].append(element)
        else:
            groups.append([element])
    return groups


def _pieces(group: list[Element], options: ChunkingOptions) -> list[_Piece]:
    size, overlap = options.chunk_size, options.chunk_overlap
    section = group[-1].section_path if group else []
    pieces: list[_Piece] = []
    buffer: list[Element] = []
    text = ""

    def flush() -> None:
        nonlocal text, buffer
        if text.strip():
            pieces.append(_Piece(text.strip(), list(buffer), section, "prose"))
        buffer, text = [], ""

    for element in group:
        if element.type in ("table", "code", "slide", "transcript"):
            flush()
            if element.type == "table":
                parts = _split_table(element.text, size)
            elif element.type == "code":
                parts = (
                    [element.text]
                    if count_tokens(element.text) <= size * 2
                    else split_to_fit(element.text, size)
                )
            else:
                parts = split_to_fit(element.text, size)
            kind = "transcript" if element.type == "transcript" else element.type
            pieces.extend(_Piece(p, [element], section, kind) for p in parts)
            continue
        if element.boundary and text and count_tokens(text) >= options.min_chunk_tokens:
            flush()  # S-02: a topic shift starts a new chunk
        unit = element.text if element.type != "list_item" else f"- {element.text}"
        if element.type in ("title", "heading"):
            continue  # headings live in the section prefix
        for part in split_to_fit(unit, size):
            candidate = f"{text}\n\n{part}".strip() if text else part
            if text and count_tokens(candidate) > size:
                previous = text
                flush()
                carry = _tail(previous, overlap)
                text = f"{carry}\n\n{part}".strip() if carry else part
            else:
                text = candidate
            buffer.append(element)
    flush()
    # C-116: merge a tiny trailing prose piece into its predecessor when it fits
    merged: list[_Piece] = []
    for piece in pieces:
        if (
            merged
            and piece.kind == "prose"
            and merged[-1].kind == "prose"
            and count_tokens(piece.text) < options.min_chunk_tokens
            and count_tokens(merged[-1].text + piece.text) <= size + options.min_chunk_tokens
        ):
            merged[-1].text += "\n\n" + piece.text
            merged[-1].elements += piece.elements
        else:
            merged.append(piece)
    return merged


def _header(title: str, section: list[str]) -> str:
    trail = [s for s in section if s and s != title]
    return " > ".join([title, *trail]) if title else " > ".join(trail)


def chunk_document(
    parsed: ParsedDocument, doc_id: str, options: ChunkingOptions | None = None
) -> list[Chunk]:
    options = options or ChunkingOptions()
    pieces = [p for group in _sections(parsed.elements) for p in _pieces(group, options)]
    chunks: list[Chunk] = []
    offset = 0
    for index, piece in enumerate(pieces):
        pages = [e.page for e in piece.elements if e.page is not None]
        header = _header(parsed.title, piece.section) if options.prepend_header else ""
        text = f"{header}\n\n{piece.text}" if header else piece.text
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{index:04d}",
                doc_id=doc_id,
                text=text,
                raw_text=piece.text,
                index=index,
                token_count=count_tokens(text),
                section_path=piece.section,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                char_start=offset,
                char_end=offset + len(piece.text),
                element_types=sorted({e.type for e in piece.elements} | {piece.kind}),
            )
        )
        offset += len(piece.text) + 2
    for previous, current in zip(chunks, chunks[1:], strict=False):  # C-118
        previous.next_id, current.prev_id = current.chunk_id, previous.chunk_id
    if options.parent_size > 0:
        chunks += _parents(chunks, doc_id, parsed.title, options.parent_size)
    return chunks


def _parents(children: list[Chunk], doc_id: str, title: str, parent_size: int) -> list[Chunk]:
    """S-01: group consecutive children of the same section into parents up to parent_size."""
    parents: list[Chunk] = []
    group: list[Chunk] = []

    def close() -> None:
        if not group:
            return
        parent_id = f"{doc_id}-p{len(parents):03d}"
        body = "\n\n".join(c.raw_text for c in group)
        header = _header(title, group[0].section_path)
        parents.append(
            Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                text=f"{header}\n\n{body}" if header else body,
                raw_text=body,
                index=group[0].index,
                token_count=count_tokens(body),
                kind="parent",
                section_path=group[0].section_path,
                page_start=group[0].page_start,
                page_end=group[-1].page_end,
            )
        )
        for child in group:
            child.parent_id = parent_id
        group.clear()

    for child in children:
        if group and (
            child.section_path != group[0].section_path
            or sum(c.token_count for c in group) + child.token_count > parent_size
        ):
            close()
        group.append(child)
    close()
    return parents

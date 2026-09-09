from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from .config import settings

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents(
    paths: list[Path],
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> list[Document]:
    """Parse supported files and return LangChain `Document` objects.

    Implementation:
        Each input path can be either a file or a folder. Folders are scanned
        recursively for supported extensions, while files are processed directly.
        Unsupported file types are skipped. Markdown/text files are read as
        UTF-8 and passed to `format_document`. PDF files are parsed page by page
        through `load_pdf_documents`, which adds page-number metadata.

    Usage:
        The ingestion graph calls this in its `parse_and_load` node. You can pass
        one or more custom file/folder paths from the ingest API or CLI.
        `uploaded_by` records who triggered this ingestion (an end-user ID
        supplied by the calling application, or its own principal when the
        caller doesn't distinguish end-users) — stored as `uploaded_by` on
        every resulting document's metadata for provenance/auditing. This is
        distinct from `authorized_users`/`authorized_teams` below, which is an
        access-control list, not an owner field; it defaults to "*" (public)
        regardless of who uploaded the document — narrowing visibility to the
        uploader is a policy decision for later, not made here.

        `org_id`, unlike `uploaded_by`, IS an access-control field: it's
        stored as `metadata["org_id"]` (default "*", i.e. unscoped/visible to
        everyone) and enforced as a hard tenant boundary at query time — see
        `rag/authorization.py::is_authorized_document`. A chunk ingested with
        a real `org_id` is only ever returned to queries from that same org.

        `account_id` works exactly the same way, for the application
        (auth-service app account) the uploader signed into: a chunk ingested
        under one account is never returned to a query made under another.

    How it helps other functions:
        This is the source of `page_content` and document-level metadata for the
        whole pipeline. Chunking, vector insertion, retrieval citations, page
        citations, metadata filters, and answer generation all depend on the
        metadata created here.
    """
    documents: list[Document] = []
    for path in paths:
        if path.is_dir():
            candidates = [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            candidates = [path]

        for file_path in candidates:
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file_path.suffix.lower() == ".pdf":
                documents.extend(
                    load_pdf_documents(
                        file_path, uploaded_by=uploaded_by, org_id=org_id, account_id=account_id
                    )
                )
            else:
                text = file_path.read_text(encoding="utf-8")
                documents.extend(
                    format_text_documents(
                        file_path,
                        text,
                        uploaded_by=uploaded_by,
                        org_id=org_id,
                        account_id=account_id,
                    )
                )
    return documents


def format_text_documents(
    path: Path,
    text: str,
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> list[Document]:
    """Create one or more text documents with section-aware metadata.

    Implementation:
        Markdown files are split into section documents by heading so each
        resulting `Document` carries a `section` value. Plain text files are
        treated as one section named after the title. The function delegates to
        `format_document` for the final metadata shape.

    Usage:
        `load_documents` calls this for `.md` and `.txt` files.

    How it helps other functions:
        Section metadata is inherited by chunks, stored in the MongoDB `chunks`
        collection, available as a retrieval filter, and shown in prompts and
        citations.
    """
    if path.suffix.lower() == ".md":
        return [
            _format_document(
                path,
                section_text,
                section=section,
                uploaded_by=uploaded_by,
                org_id=org_id,
                account_id=account_id,
            )
            for section, section_text in _split_markdown_sections(text)
        ]
    return [
        _format_document(
            path,
            text,
            section=_extract_title(text) or path.stem,
            uploaded_by=uploaded_by,
            org_id=org_id,
            account_id=account_id,
        )
    ]


def format_document(
    path: Path,
    text: str,
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> Document:
    """Create a normalized LangChain `Document` from raw file text.

    Implementation:
        The function extracts a title, computes a SHA-256 content hash, records
        source information, optional section/page/layout fields, authorization
        defaults, and stores stripped text as `page_content`. The hash can later
        support deduplication, stale content detection, deletion, and re-indexing.

    Usage:
        `load_documents` calls this once per supported file. It can also be used
        directly in tests when you want to build a document from an in-memory
        string.

    How it helps other functions:
        The metadata created here is inherited by every chunk, written to the
        MongoDB `chunks` collection, used by hybrid retrieval for identity,
        metadata filters, parent-child expansion, and displayed in prompts for
        citations.
    """
    return _format_document(
        path, text, uploaded_by=uploaded_by, org_id=org_id, account_id=account_id
    )


def _format_document(
    path: Path,
    text: str,
    section: str | None = None,
    page_number: int | None = None,
    extra_metadata: dict[str, object] | None = None,
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> Document:
    """Build the final LangChain document object with optional rich metadata.

    Implementation:
        The helper centralizes document creation for Markdown, text, and PDF
        inputs. It extracts title, computes a content hash for the specific
        section or page text, adds optional `section` and `page_number` fields,
        applies default authorization metadata, and merges parser-specific
        metadata such as table or figure counts.

    Usage:
        `format_document`, `format_text_documents`, and `load_pdf_documents` call
        this helper so all document types have the same metadata contract.

    How it helps other functions:
        Consistent metadata lets chunking, vector-store insertion,
        metadata-filtered retrieval, authorization filtering, citations, and
        evaluation work the same way across file types.
    """
    title = _extract_title(text) or path.stem.replace("_", " ").replace("-", " ").title()
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    metadata = {
        "source": str(path),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "title": title,
        "section": section or title,
        "content_hash": content_hash,
        "loaded_at": datetime.now(UTC).isoformat(),
        # Provenance — who uploaded/triggered ingestion of this document.
        # Not an access-control field; see the module docstring above.
        "uploaded_by": uploaded_by or "",
        # Tenant boundary — enforced (unlike uploaded_by) at query time by
        # rag/authorization.py. "*" means unscoped/visible to every org.
        "org_id": org_id or "*",
        # Application boundary — enforced the same way as org_id. "*" means
        # visible to every application.
        "account_id": account_id or "*",
        "authorized_users": "*",
        "authorized_teams": "*",
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(page_content=text.strip(), metadata=metadata)


def load_pdf_documents(
    path: Path,
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> list[Document]:
    """Parse a PDF into page-level documents with `page_number` metadata.

    Implementation:
        The function first tries richer layout extraction through
        `load_pdf_documents_with_layout` when enabled. If that does not return
        content, it falls back to plain `pypdf` text extraction and optionally OCR
        for scanned pages. Page numbers are 1-based so they match user-facing
        PDF citations.

    Usage:
        `load_documents` calls this automatically for `.pdf` files.

    How it helps other functions:
        Page-level documents allow chunks, retrieval results, prompts, and
        citations to point back to exact PDF pages, while layout metadata helps
        users understand whether evidence came from text, tables, figures, or
        OCR.
    """
    if settings.enable_pdf_layout_extraction:
        layout_documents = load_pdf_documents_with_layout(
            path, uploaded_by=uploaded_by, org_id=org_id, account_id=account_id
        )
        if layout_documents:
            return layout_documents

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to ingest PDF files: pip install pypdf") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        extraction_method = "pypdf"
        if not text and settings.enable_pdf_ocr:
            text = ocr_pdf_page(path, index)
            extraction_method = "ocr" if text else extraction_method
        if not text:
            continue
        documents.append(
            _format_document(
                path,
                text,
                section=f"Page {index}",
                page_number=index,
                extra_metadata={
                    "extraction_method": extraction_method,
                    "table_count": 0,
                    "figure_count": 0,
                },
                uploaded_by=uploaded_by,
                org_id=org_id,
                account_id=account_id,
            )
        )
    return documents


def load_pdf_documents_with_layout(
    path: Path,
    uploaded_by: str | None = None,
    org_id: str | None = None,
    account_id: str | None = None,
) -> list[Document]:
    """Parse PDF pages with layout-aware text, table, and figure metadata.

    Implementation:
        The function lazily imports `pdfplumber`, extracts page text, extracts
        detected tables into Markdown, counts image objects as figure candidates,
        and creates one `Document` per page. If a page has no extractable text
        and OCR is enabled, it calls `ocr_pdf_page`.

    Usage:
        `load_pdf_documents` calls this first when
        `ENABLE_PDF_LAYOUT_EXTRACTION=true`.

    How it helps other functions:
        Richer page content lets chunking and retrieval include table evidence,
        while metadata such as `table_count`, `figure_count`, and
        `extraction_method` improves citations, debugging, and evaluation.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    documents: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            tables = page.extract_tables() or []
            table_text = "\n\n".join(_table_to_markdown(table) for table in tables if table)
            figure_count = len(getattr(page, "images", []) or [])
            extraction_method = "pdfplumber"

            combined_parts = [part for part in [text, table_text] if part.strip()]
            combined_text = "\n\n".join(combined_parts).strip()
            if not combined_text and settings.enable_pdf_ocr:
                combined_text = ocr_pdf_page(path, index)
                extraction_method = "ocr" if combined_text else extraction_method

            if not combined_text:
                continue

            documents.append(
                _format_document(
                    path,
                    combined_text,
                    section=f"Page {index}",
                    page_number=index,
                    extra_metadata={
                        "extraction_method": extraction_method,
                        "table_count": len(tables),
                        "figure_count": figure_count,
                        "has_tables": bool(tables),
                        "has_figures": figure_count > 0,
                    },
                    uploaded_by=uploaded_by,
                    org_id=org_id,
                    account_id=account_id,
                )
            )
    return documents


def _table_to_markdown(table: list[list[object]]) -> str:
    """Serialize a PDF table into Markdown for retrieval.

    Implementation:
        The first row is treated as the header when present. Empty cells are
        normalized to empty strings. A separator row is added so the table is
        readable as Markdown and can be embedded as text.

    Usage:
        `load_pdf_documents_with_layout` calls this for every table detected by
        `pdfplumber`.

    How it helps other functions:
        Turning tables into text allows chunking, embeddings, keyword search,
        and generation to use tabular evidence instead of losing it during PDF
        parsing.
    """
    rows = [
        ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
        for row in table
    ]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] or [["" for _ in header]]
    separator = ["---" for _ in header]
    markdown_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)


def ocr_pdf_page(path: Path, page_number: int) -> str:
    """Extract text from a scanned PDF page with OCR when local tools exist.

    Implementation:
        The function lazily imports `pdf2image` and `pytesseract`, renders the
        requested page as an image, and runs OCR over the rendered image. If the
        optional tools or system binaries are unavailable, it returns an empty
        string so normal ingestion can continue.

    Usage:
        PDF parsing calls this only when normal text/layout extraction finds no
        content and `ENABLE_PDF_OCR=true`.

    How it helps other functions:
        OCR makes scanned pages available to the same chunking, embedding,
        retrieval, citation, and evaluation flow as text-native PDFs.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return ""

    try:
        images = convert_from_path(
            str(path),
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0]).strip()
    except Exception:
        return ""


def _extract_title(text: str) -> str | None:
    """Extract the first Markdown H1 heading from a document.

    Implementation:
        The function scans lines until it finds one starting with `# ` and
        returns the cleaned heading text. If no heading exists, it returns
        `None` so `format_document` can fall back to the file name.

    Usage:
        This is an internal helper for `format_document`.

    How it helps other functions:
        A useful title improves metadata quality. Prompt construction and local
        reranking use title metadata to make retrieved evidence easier to read
        and slightly easier to prioritize.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return None


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown content into heading-aware section documents.

    Implementation:
        The function scans the document line by line, tracks the latest Markdown
        heading, and groups following lines under that heading. Content before
        the first heading is grouped under the extracted title or `Introduction`.

    Usage:
        `format_text_documents` calls this for `.md` files before document
        metadata is created.

    How it helps other functions:
        Section-specific documents make chunk metadata more precise, allow
        `section` metadata filters, and improve citations because retrieved
        chunks can identify the part of the source document they came from.
    """
    sections: list[tuple[str, list[str]]] = []
    current_heading = _extract_title(text) or "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = _markdown_heading(line)
        if heading:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = heading
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [
        (heading, "\n".join(lines).strip())
        for heading, lines in sections
        if "\n".join(lines).strip()
    ]


def _markdown_heading(line: str) -> str | None:
    """Return clean Markdown heading text from a single line.

    Implementation:
        Matches ATX-style headings from `#` through `######`, strips the heading
        marks, and returns the readable heading text.

    Usage:
        `_split_markdown_sections` uses this while scanning Markdown files.

    How it helps other functions:
        Clean section names become metadata values that can be filtered during
        retrieval and shown in citations.
    """
    match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
    if not match:
        return None
    return match.group(1).strip()

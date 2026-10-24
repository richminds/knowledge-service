"""File-type detection from content (magic bytes), not the extension (C-069)."""

import io
import zipfile

# kind -> canonical extension used by the parser router
_OOXML = {
    "word/": ".docx",
    "ppt/": ".pptx",
    "xl/": ".xlsx",
}


# Image formats: name -> (MIME type, Document Intelligence accepts it, vision model accepts it)
IMAGE_FORMATS: dict[str, tuple[str, bool, bool]] = {
    "png": ("image/png", True, True),
    "jpeg": ("image/jpeg", True, True),
    "gif": ("image/gif", False, True),
    "webp": ("image/webp", False, True),
    "bmp": ("image/bmp", True, False),
    "tiff": ("image/tiff", True, False),
    "heif": ("image/heif", True, False),
    "avif": ("image/avif", False, False),
}
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}


def image_format(content: bytes) -> str:
    """Return the IMAGE_FORMATS key for image content, or "" if it is not an image."""
    if content.startswith(b"\x89PNG"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"GIF8"):
        return "gif"
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "webp"
    # BMP: "BM", file size, then 4 reserved zero bytes (avoids matching text starting "BM")
    if content.startswith(b"BM") and len(content) > 26 and content[6:10] == b"\x00" * 4:
        return "bmp"
    if content[4:8] == b"ftyp":
        brand = content[8:12]
        if brand in _HEIF_BRANDS:
            return "heif"
        if brand in (b"avif", b"avis"):
            return "avif"
    return ""


def detect_type(content: bytes, name: str = "") -> str:
    """Return a canonical extension (".pdf", ".docx", ".md", ...) or "" if unknown.

    Content signature wins; the file name only disambiguates text formats.
    """
    head = content[:8]
    lowered_name = name.lower()
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith((b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"\xcf\xfa\xed\xfe")):
        return ".exe"  # executables are never parsed
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            return ""
        for prefix, ext in _OOXML.items():
            if any(n.startswith(prefix) for n in names):
                return ext
        return ".zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2: legacy .doc/.xls/.ppt/.msg
        return ".msg" if lowered_name.endswith(".msg") else ".ole"
    if image_format(content):  # before media: WebP is RIFF and HEIF is ftyp, like audio/video
        return ".image"
    if head[4:8] == b"ftyp" or head.startswith((b"ID3", b"RIFF", b"OggS", b"fLaC")):
        return ".media"  # mp4/m4a/mp3/wav/ogg/flac -> transcription (C-082)
    # Text formats: must decode as UTF-8 (after an optional BOM)
    try:
        text = content[:4096].decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content[:4096].decode("utf-16")
        except UnicodeDecodeError:
            return ""
    sniff = text.lstrip().lower()
    if sniff.startswith(("<!doctype html", "<html")) or "<body" in sniff[:2000]:
        return ".html"
    header_block = sniff.split("\n\n", 1)[0]
    email_headers = (
        "from:",
        "to:",
        "subject:",
        "date:",
        "mime-version:",
        "received:",
        "return-path:",
    )
    if (
        lowered_name.endswith(".eml")
        or sum(line.startswith(email_headers) for line in header_block.splitlines()) >= 2
    ):
        return ".eml"
    for ext in (".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml"):
        if lowered_name.endswith(ext):
            return ".md" if ext == ".markdown" else ext
    for ext in (".py", ".js", ".ts", ".java", ".cs", ".go", ".sql", ".sh", ".ps1", ".r"):
        if lowered_name.endswith(ext):
            return ext
    return ".txt"


# Upload allow-list check: what detect_type() must return for each allowed extension.
_IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml"}
_BINARY_TYPES = {".pdf", ".docx", ".pptx", ".xlsx", ".zip", ".exe", ".ole", ".msg", ".image"}
_BINARY_TYPES |= {".media", ""}


def upload_problem(name: str, content: bytes, allowed: list[str]) -> str | None:
    """Why an upload is refused under the `allowed` extensions, or None if it is accepted.

    The extension must be allowed and the content must really be that kind of file, so a
    renamed executable or archive is refused even with an allowed name."""
    dot = name.rfind(".")
    ext = name[dot:].lower() if dot > 0 else ""
    if ext not in allowed:
        return f"Unsupported file type {ext or '(none)'}; allowed: {', '.join(allowed)}"
    detected = detect_type(content, name)
    if ext in _IMAGE_EXTENSIONS:
        ok = detected == ".image" and image_format(content) == _IMAGE_EXTENSIONS[ext]
    elif ext in _TEXT_EXTENSIONS:
        ok = detected not in _BINARY_TYPES
    else:
        ok = detected == ext
    return None if ok else f"Content does not match the {ext} extension"


def safe_relative_path(path: str, fallback: str) -> str:
    """A client-supplied folder path (e.g. `policies/hr/leave.pdf`) made safe for use in a
    source id: no drive, root, `.` or `..` segments, no control characters. Falls back to the
    file name when nothing usable is left."""
    parts = []
    for part in path.replace("\\", "/").split("/"):
        part = "".join(ch for ch in part if ch.isprintable()).strip()
        if part and part not in (".", "..") and not part.endswith(":"):
            parts.append(part[:200])
    return "/".join(parts[-12:]) or fallback

"""Images and audio/video (C-075, C-078): formats are still detected from content, but the LLM
gateway is text-only, so they are refused with a clear error rather than half-ingested."""

import io
import struct

import pytest

from app.core.config import Settings
from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.filetypes import detect_type, image_format
from app.rag.ingestion.models import SourceDocument
from app.rag.ingestion.parsing import SUPPORTED_EXTENSIONS, ParserRouter

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 32
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32
BMP = b"BM" + struct.pack("<I", 70) + b"\x00" * 4 + struct.pack("<I", 54) + b"\x00" * 56
WAV = b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 32
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32


def _doc(content: bytes, name: str) -> SourceDocument:
    return SourceDocument(source=f"test/{name}", name=name, content=content)


@pytest.mark.parametrize(
    ("content", "fmt", "kind"),
    [
        (PNG, "png", ".image"),
        (GIF, "gif", ".image"),
        (WEBP, "webp", ".image"),  # RIFF container, but not audio
        (HEIC, "heif", ".image"),  # ftyp box, but not video
        (BMP, "bmp", ".image"),
        (WAV, "", ".media"),
        (MP4, "", ".media"),
        (b"BMW fleet policy: cars are leased for three years.", "", ".txt"),
    ],
)
def test_image_formats_are_detected_from_content(content, fmt, kind):
    assert image_format(content) == fmt
    assert detect_type(content, "file.bin") == kind


@pytest.mark.parametrize(
    ("content", "name"),
    [(PNG, "scan.png"), (HEIC, "photo.heic"), (WAV, "call.wav"), (MP4, "town-hall.mp4")],
)
async def test_images_and_media_are_refused(content, name):
    with pytest.raises(DocumentLoadError, match="images and audio/video are not ingested"):
        await ParserRouter().parse(_doc(content, name))


def test_image_and_audio_extensions_are_not_uploadable():
    for ext in (".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4"):
        assert ext not in SUPPORTED_EXTENSIONS
    with pytest.raises(ValueError, match="does not support .png"):
        Settings(ingestion_upload_extensions=".pdf,.png")
    assert ".png" not in Settings.model_fields["ingestion_upload_extensions"].get_default(
        call_default_factory=True
    )


async def test_scanned_pdf_pages_are_flagged_not_described():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # a page with no text layer
    buffer = io.BytesIO()
    writer.write(buffer)

    parsed = await ParserRouter().parse(_doc(buffer.getvalue(), "scan.pdf"))

    assert parsed.parser == "pypdf"
    assert "pages_without_text:1/1" in parsed.quality_flags

"""Image ingestion (C-075, C-078): format detection and routing between Document Intelligence
(OCR) and the vision model (description). Both services are fakes; nothing calls Azure."""

import struct

import pytest

from app.core.config import Settings
from app.core.exceptions import DocumentLoadError
from app.rag.ingestion.filetypes import detect_type, image_format
from app.rag.ingestion.models import Element, ParsedDocument, SourceDocument
from app.rag.ingestion.parsing import ParserRouter

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 32
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32
BMP = b"BM" + struct.pack("<I", 70) + b"\x00" * 4 + struct.pack("<I", 54) + b"\x00" * 56


class FakeDocIntel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def parse_image(self, source: str, name: str, content: bytes) -> ParsedDocument:
        self.calls += 1
        elements = [Element("paragraph", self.text)] if self.text else []
        return ParsedDocument(source, name, elements, "docintel:prebuilt-layout")


class FakeVision:
    def __init__(self) -> None:
        self.calls = 0

    async def parse(self, source: str, name: str, content: bytes) -> ParsedDocument:
        self.calls += 1
        return ParsedDocument(
            source, name, [Element("figure", "Image description: an org chart")], "vision"
        )


def _image(content: bytes, name: str = "scan.png") -> SourceDocument:
    return SourceDocument(source=f"test/{name}", name=name, content=content)


SCANNED_TEXT = "Expense claims must be submitted within 30 days of travel."


@pytest.mark.parametrize(
    ("content", "fmt", "kind"),
    [
        (PNG, "png", ".image"),
        (GIF, "gif", ".image"),
        (WEBP, "webp", ".image"),  # RIFF container, but not audio
        (HEIC, "heif", ".image"),  # ftyp box, but not video
        (BMP, "bmp", ".image"),
        (b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 32, "", ".media"),
        (b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32, "", ".media"),
        (b"BMW fleet policy: cars are leased for three years.", "", ".txt"),
    ],
)
def test_image_formats_are_detected_from_content(content, fmt, kind):
    assert image_format(content) == fmt
    assert detect_type(content, "file.bin") == kind


async def test_auto_uses_ocr_when_the_image_has_text():
    docintel, vision = FakeDocIntel(SCANNED_TEXT), FakeVision()
    parsed = await ParserRouter(docintel, vision).parse(_image(PNG))
    assert parsed.parser == "docintel:prebuilt-layout"
    assert SCANNED_TEXT in parsed.text
    assert vision.calls == 0


async def test_auto_adds_a_description_when_ocr_finds_little_text():
    docintel, vision = FakeDocIntel("Fig 1"), FakeVision()
    parsed = await ParserRouter(docintel, vision).parse(_image(PNG))
    assert parsed.parser == "docintel:prebuilt-layout+vision"
    assert "Fig 1" in parsed.text and "org chart" in parsed.text


async def test_vision_only_formats_skip_document_intelligence():
    docintel, vision = FakeDocIntel(SCANNED_TEXT), FakeVision()
    parsed = await ParserRouter(docintel, vision).parse(_image(WEBP, "diagram.webp"))
    assert parsed.parser == "vision" and docintel.calls == 0


async def test_ocr_only_formats_work_without_vision():
    parsed = await ParserRouter(FakeDocIntel(SCANNED_TEXT)).parse(_image(HEIC, "photo.heic"))
    assert SCANNED_TEXT in parsed.text


async def test_docintel_mode_never_calls_vision():
    docintel, vision = FakeDocIntel("x"), FakeVision()
    router = ParserRouter(docintel, vision, image_parser="docintel")
    parsed = await router.parse(_image(PNG))
    assert parsed.parser == "docintel:prebuilt-layout" and vision.calls == 0


async def test_docintel_plus_vision_always_combines():
    router = ParserRouter(FakeDocIntel(SCANNED_TEXT), FakeVision(), image_parser="docintel+vision")
    parsed = await router.parse(_image(PNG))
    assert SCANNED_TEXT in parsed.text and "org chart" in parsed.text


async def test_off_and_unsupported_images_fail_clearly():
    with pytest.raises(DocumentLoadError, match="IMAGE_PARSER=off"):
        await ParserRouter(FakeDocIntel("x"), image_parser="off").parse(_image(PNG))
    with pytest.raises(DocumentLoadError, match="No image processor for gif"):
        await ParserRouter(FakeDocIntel(SCANNED_TEXT)).parse(_image(GIF, "anim.gif"))
    with pytest.raises(DocumentLoadError, match="DOC_INTELLIGENCE_ENDPOINT"):
        await ParserRouter().parse(_image(PNG))


def test_image_parser_settings_are_validated():
    settings = Settings(image_parser="docintel", doc_intelligence_endpoint="")
    assert any("DOC_INTELLIGENCE_ENDPOINT" in e for e in settings.configuration_errors())
    settings = Settings(image_parser="vision", azure_openai_vision_deployment="")
    assert any("AZURE_OPENAI_VISION_DEPLOYMENT" in e for e in settings.configuration_errors())

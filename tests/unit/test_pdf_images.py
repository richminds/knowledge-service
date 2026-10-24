"""Images inside natively parsed PDFs (C-075, C-078) and the vision call for reasoning models.

PDFs are built in memory with Pillow; the vision model is a fake, so nothing calls Azure."""

import io
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError
from PIL import Image

from app.rag.ingestion.media import VisionDescriber
from app.rag.ingestion.models import SourceDocument
from app.rag.ingestion.parsing import ParserRouter, extract_pdf_images


def _pdf(*pages: Image.Image) -> bytes:
    """An image-only PDF, one picture per page (what a scanned page or pasted diagram is)."""
    buffer = io.BytesIO()
    pages[0].save(buffer, "PDF", save_all=True, append_images=list(pages[1:]))
    return buffer.getvalue()


def _picture(size=(400, 300), colour=(30, 90, 160)) -> Image.Image:
    return Image.new("RGB", size, colour)


def _doc(content: bytes) -> SourceDocument:
    return SourceDocument(source="test/arch.pdf", name="arch.pdf", content=content)


class FakeVision:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def describe(self, content: bytes) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("vision unavailable")
        return "Gateway Server connects to port 5480 of the AMS EC2 host."


async def test_image_only_pdf_is_described_instead_of_left_empty():
    vision = FakeVision()
    parsed = await ParserRouter(vision=vision).parse(_doc(_pdf(_picture())))
    assert parsed.parser == "pypdf+vision"
    assert "Figure (page 1): Gateway Server connects to port 5480" in parsed.text
    assert parsed.elements[0].page == 1
    assert not any(f.startswith("pages_without_text") for f in parsed.quality_flags)


async def test_icons_and_repeated_images_are_described_once():
    logo = _picture((40, 40), (200, 0, 0))
    pdf = _pdf(_picture(), _picture(), logo)  # same diagram twice, then a small logo
    vision = FakeVision()
    parsed = await ParserRouter(vision=vision).parse(_doc(pdf))
    assert vision.calls == 1
    assert [e.page for e in parsed.elements if e.type == "figure"] == [1]


async def test_images_beyond_the_limit_are_flagged():
    pdf = _pdf(*(_picture(colour=(i * 40, 0, 0)) for i in range(3)))
    vision = FakeVision()
    parsed = await ParserRouter(vision=vision, pdf_max_images=2).parse(_doc(pdf))
    assert vision.calls == 2
    assert "pdf_images_not_described:1" in parsed.quality_flags


async def test_a_failed_description_keeps_the_pdf_and_flags_it():
    parsed = await ParserRouter(vision=FakeVision(fail=True)).parse(_doc(_pdf(_picture())))
    assert parsed.parser == "pypdf"
    assert "pdf_image_descriptions_failed:1" in parsed.quality_flags
    assert "pages_without_text:1/1" in parsed.quality_flags


@pytest.mark.parametrize(
    "router",
    [
        ParserRouter(),  # no vision deployment
        ParserRouter(vision=FakeVision(), pdf_image_descriptions=False),
        ParserRouter(vision=FakeVision(), image_parser="off"),
    ],
)
async def test_pdf_images_are_skipped_when_disabled(router):
    parsed = await router.parse(_doc(_pdf(_picture())))
    assert parsed.parser == "pypdf"
    assert "pages_without_text:1/1" in parsed.quality_flags


def test_large_and_non_rgb_images_are_sent_as_bounded_png():
    pdf = _pdf(_picture((3000, 1000)), Image.new("CMYK", (300, 300)))
    images, over = extract_pdf_images(pdf, min_pixels=150, max_images=10)
    assert over == 0 and [page for page, _ in images] == [1, 2]
    for _, data in images:
        picture = Image.open(io.BytesIO(data))
        assert max(picture.size) <= 2048 and picture.mode in ("RGB", "L")


# -- vision call parameters ---------------------------------------------------------------------
class FakeCompletions:
    def __init__(self, reject: str | None = None) -> None:
        self.calls: list[dict] = []
        self.reject = reject

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject and self.reject in kwargs:
            request = httpx.Request("POST", "https://example.openai.azure.com")
            raise BadRequestError(
                "unsupported parameter",
                response=httpx.Response(400, request=request),
                body={"param": self.reject},
            )
        message = SimpleNamespace(content="a diagram")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _describer(deployment: str, completions: FakeCompletions) -> VisionDescriber:
    describer = VisionDescriber("https://example.openai.azure.com", "key", "2024-10-21", deployment)
    describer._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return describer


async def test_reasoning_deployments_get_max_completion_tokens_and_no_temperature():
    completions = FakeCompletions()
    assert await _describer("gpt-5.5", completions).describe(b"\x89PNG") == "a diagram"
    sent = completions.calls[0]
    assert "temperature" not in sent and "max_tokens" not in sent
    assert sent["max_completion_tokens"] > 1500


async def test_other_deployments_keep_max_tokens_and_temperature():
    completions = FakeCompletions()
    await _describer("gpt-4o", completions).describe(b"\x89PNG")
    assert completions.calls[0]["max_tokens"] == 1500 and completions.calls[0]["temperature"] == 0


async def test_a_custom_named_reasoning_deployment_is_learned_from_the_rejection():
    completions = FakeCompletions(reject="temperature")
    describer = _describer("diagram-reader", completions)
    assert await describer.describe(b"\x89PNG") == "a diagram"
    await describer.describe(b"\x89PNG")
    assert len(completions.calls) == 3  # rejected once, then reasoning parameters every time
    assert "temperature" not in completions.calls[2]

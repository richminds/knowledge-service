"""Images and audio/video (C-078, C-082).

- VisionDescriber: describes charts/diagrams/photos with a vision-capable chat deployment
  (e.g. gpt-4o or gpt-5.x) so image-only content becomes searchable: image files, and the
  images embedded in natively parsed PDFs (PDF_IMAGE_DESCRIPTIONS). Placeholder:
  AZURE_OPENAI_VISION_DEPLOYMENT (blank = images without OCR text are skipped).
- Transcriber: Azure OpenAI Whisper (AZURE_OPENAI_WHISPER_DEPLOYMENT) with timestamps.
  Files over 25 MB, or speaker labels (diarisation), need Azure AI Speech batch
  transcription - TODO: SPEECH_ENDPOINT.
"""

import base64
import io

from openai import BadRequestError
from openai.types.chat import ChatCompletionMessageParam

from app.core.azure_auth import build_openai_client
from app.core.exceptions import DocumentLoadError
from app.core.usage import record_usage
from app.rag.generation.llm_client import is_reasoning_model, reasoning_params
from app.rag.ingestion.models import Element, ParsedDocument

VISION_PROMPT = (
    "Describe this image for a search index. Transcribe any visible text. For charts, give the "
    "title, axes, series and the key values or trends. For diagrams, list the components and how "
    "they connect. Be factual and concise; do not speculate."
)
_WHISPER_LIMIT = 25 * 1024 * 1024


def _mime(content: bytes) -> str:
    from app.rag.ingestion.filetypes import IMAGE_FORMATS, image_format

    return IMAGE_FORMATS.get(image_format(content), ("image/jpeg",))[0]


class VisionDescriber:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        max_tokens: int = 1500,
    ) -> None:
        self._client = build_openai_client(endpoint, api_key, api_version)
        self._deployment = deployment
        self._max_tokens = max_tokens  # architecture diagrams need ~1,500-2,000 tokens
        # Reasoning deployments (gpt-5.x, o-series) reject max_tokens and temperature. Known by
        # name, or learned from the first rejection for a deployment with a custom name.
        self._reasoning = is_reasoning_model(deployment)

    async def describe(self, content: bytes) -> str:
        data_url = f"data:{_mime(content)};base64,{base64.b64encode(content).decode()}"
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        params = {"max_tokens": self._max_tokens, "temperature": 0}
        try:
            response = await self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                **(reasoning_params(params) if self._reasoning else params),
            )
        except BadRequestError as exc:
            if self._reasoning or getattr(exc, "param", None) not in ("max_tokens", "temperature"):
                raise
            self._reasoning = True
            response = await self._client.chat.completions.create(
                model=self._deployment, messages=messages, **reasoning_params(params)
            )
        if response.usage:
            record_usage(
                "figure_caption",
                self._deployment,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        return response.choices[0].message.content or ""

    async def parse(self, source: str, name: str, content: bytes) -> ParsedDocument:
        text = await self.describe(content)
        return ParsedDocument(
            source=source,
            title=name.rsplit(".", 1)[0],
            elements=[Element("figure", f"Image description: {text}")],
            parser="vision",
        )


class Transcriber:
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str) -> None:
        self._client = build_openai_client(endpoint, api_key, api_version)
        self._deployment = deployment

    async def parse(self, source: str, name: str, content: bytes) -> ParsedDocument:
        if len(content) > _WHISPER_LIMIT:
            raise DocumentLoadError("Media over 25 MB needs Azure AI Speech batch transcription")
        audio = io.BytesIO(content)
        audio.name = name
        result = await self._client.audio.transcriptions.create(
            model=self._deployment, file=audio, response_format="verbose_json"
        )
        elements = []
        for segment in getattr(result, "segments", None) or []:
            start = int(getattr(segment, "start", 0))
            stamp = f"[{start // 3600:02d}:{start % 3600 // 60:02d}:{start % 60:02d}]"
            elements.append(Element("transcript", f"{stamp} {segment.text.strip()}"))
        if not elements:
            elements = [Element("transcript", getattr(result, "text", ""))]
        return ParsedDocument(
            source=source, title=name.rsplit(".", 1)[0], elements=elements, parser="whisper"
        )

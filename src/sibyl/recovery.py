"""One-page recovery orchestration and provider boundaries."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from sibyl.experiments.trocr import Recognizer, TrocrRecognizer

VLM_MAX_DIMENSIONS = (1536, 2048)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_QWEN_MODEL = "qwen3-vl:8b"


@dataclass(frozen=True)
class RegionBounds:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class RegionCandidate:
    value: str
    evidence: str | None = None
    status: str = "unresolved"


@dataclass(frozen=True)
class RecoveredRegion:
    order: int
    kind: str
    bounds: RegionBounds
    qwen_text: str | None
    text: str
    candidates: list[RegionCandidate] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    recognizer: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredPage:
    source: dict[str, Any]
    dimensions: dict[str, int]
    interpretation: dict[str, Any]
    regions: list[RecoveredRegion]
    runtime: dict[str, Any]


class PageInterpreter(Protocol):
    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


def _bounded_dimensions(size: tuple[int, int], maximum: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    scale = min(maximum[0] / width, maximum[1] / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_vlm_image(source: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Create the deterministic, in-memory grayscale derivative for the VLM."""
    image = source.convert("L")
    dimensions = _bounded_dimensions(image.size, VLM_MAX_DIMENSIONS)
    if dimensions != image.size:
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
    return image, dimensions


def _image_data(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    import base64

    return base64.b64encode(output.getvalue()).decode("ascii")


class OllamaPageInterpreter:
    """Qwen3-VL adapter using Ollama's local chat API and JSON schema output."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        schema = {
            "type": "object",
            "properties": {
                "page_interpretation": {"type": "object"},
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "order": {"type": "integer"},
                            "left": {"type": "number"},
                            "top": {"type": "number"},
                            "right": {"type": "number"},
                            "bottom": {"type": "number"},
                            "text": {"type": "string"},
                        },
                        "required": ["kind", "order", "left", "top", "right", "bottom"],
                    },
                },
            },
            "required": ["page_interpretation", "regions"],
        }
        prompt = (
            "Interpret this handwritten page. Return only JSON matching the schema. "
            "Coordinates are normalized 0..1 relative to the supplied image. Identify "
            "each text region in reading order, preserve uncertain wording, and include "
            "drawings or diagrams in page_interpretation without inventing text."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": 0,
            "messages": [{"role": "user", "content": prompt, "images": [_image_data(image)]}],
        }
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unable to query Ollama/Qwen ({self.model}): {error}") from error
        message = body.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama/Qwen returned no structured message content")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama/Qwen returned invalid structured JSON") from error
        if not isinstance(result, dict) or not isinstance(result.get("regions"), list):
            raise RuntimeError("Ollama/Qwen response lacks a structured regions list")
        return result, (time.perf_counter() - started) * 1000

    def release(self) -> None:
        # keep_alive=0 on the interpretation request asks Ollama to release the model.
        return None


def _bounds(item: dict[str, Any], source_size: tuple[int, int]) -> RegionBounds:
    width, height = source_size
    values = [float(item.get(key, 0)) for key in ("left", "top", "right", "bottom")]
    left, top, right, bottom = [
        round(value * dimension)
        for value, dimension in zip(values, (width, height, width, height), strict=True)
    ]
    left, right = sorted((max(0, min(width, left)), max(0, min(width, right))))
    top, bottom = sorted((max(0, min(height, top)), max(0, min(height, bottom))))
    if right <= left or bottom <= top:
        raise ValueError("Qwen returned an empty text region")
    return RegionBounds(left, top, right, bottom)


def recover_page(
    image_path: Path,
    interpreter: PageInterpreter | None = None,
    recognizer: Recognizer | None = None,
    *,
    recognizer_metadata: dict[str, Any] | None = None,
) -> RecoveredPage:
    """Interpret one source page, then recognize its identified regions."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error

    vlm_image, vlm_dimensions = prepare_vlm_image(source)
    interpreter = interpreter or OllamaPageInterpreter()
    interpretation, vlm_ms = interpreter.interpret(vlm_image)
    interpreter.release()
    if recognizer is None:
        recognizer, load_ms, cuda, device, gpu = TrocrRecognizer.from_local_cache()
        recognizer_metadata = {
            "model_load_ms": load_ms,
            "cuda_available": cuda,
            "device": device,
            "gpu": gpu,
        }
    recognizer_metadata = recognizer_metadata or {}
    regions: list[RecoveredRegion] = []
    for raw in sorted(interpretation["regions"], key=lambda item: int(item.get("order", 0))):
        bounds = _bounds(raw, source.size)
        started = time.perf_counter()
        region_image = source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert(
            "RGB"
        )
        text, inference_ms = recognizer.recognize(region_image)
        elapsed_ms = (time.perf_counter() - started) * 1000
        regions.append(
            RecoveredRegion(
                order=int(raw.get("order", len(regions))),
                kind=str(raw.get("kind", "text")),
                bounds=bounds,
                qwen_text=raw.get("text"),
                text=text,
                source={"image": str(image_path), "bounds": asdict(bounds)},
                recognizer={
                    "inference_ms": round(inference_ms, 3),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
            )
        )
    runtime = {
        "vlm_model": getattr(interpreter, "model", None),
        "vlm_ms": round(vlm_ms, 3),
        "vlm_dimensions": {"width": vlm_dimensions[0], "height": vlm_dimensions[1]},
        "recognizer": recognizer_metadata,
    }
    return RecoveredPage(
        source={"image": str(image_path)},
        dimensions={"width": source.width, "height": source.height},
        interpretation=interpretation.get("page_interpretation", {}),
        regions=regions,
        runtime=runtime,
    )


def format_recovery(page: RecoveredPage) -> str:
    return json.dumps(asdict(page), indent=2)

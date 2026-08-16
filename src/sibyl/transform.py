"""One-page transform orchestration and provider boundaries."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

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
class PreparedVlmImage:
    image: Image.Image
    source_dimensions: tuple[int, int]
    prepared_dimensions: tuple[int, int]
    scale: float
    preparation_ms: float


@dataclass(frozen=True)
class RegionCandidate:
    value: str
    evidence: str | None = None
    status: str = "unresolved"


@dataclass(frozen=True)
class TransformedRegion:
    order: int
    kind: str
    bounds: RegionBounds
    prepared_bounds: RegionBounds | None
    qwen_text: str | None
    text: str
    normalized_bounds: tuple[float, float, float, float] | None = None
    candidates: list[RegionCandidate] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    recognizer: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformedPage:
    source: dict[str, Any]
    dimensions: dict[str, int]
    interpretation: dict[str, Any]
    regions: list[TransformedRegion]
    runtime: dict[str, Any]
    page_text: list[str] = field(default_factory=list)


class PageInterpreter(Protocol):
    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


class DrawingLocalizer(Protocol):
    model: str

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


TEXT_REGION_PADDING = 0.02


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


def prepare_vlm_image_with_metadata(source: Image.Image) -> PreparedVlmImage:
    """Create the deterministic inference representation and measure preparation."""
    started = time.perf_counter()
    image, dimensions = prepare_vlm_image(source)
    scale = min(
        dimensions[0] / source.width,
        dimensions[1] / source.height,
    )
    return PreparedVlmImage(
        image=image,
        source_dimensions=source.size,
        prepared_dimensions=dimensions,
        scale=scale,
        preparation_ms=(time.perf_counter() - started) * 1000,
    )


def _image_data(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    import base64

    return base64.b64encode(output.getvalue()).decode("ascii")


def _valid_text_region_list(regions: Any) -> bool:
    if not isinstance(regions, list):
        return False
    return all(
        isinstance(region, dict)
        and isinstance(region.get("bbox_2d"), list)
        and _valid_qwen_bbox(region["bbox_2d"])
        and (region.get("order") is None or isinstance(region.get("order"), int))
        and (region.get("kind") is None or region.get("kind") == "text")
        and (region.get("text") is None or isinstance(region.get("text"), str))
        for region in regions
    )


class OllamaPageInterpreter:
    """Qwen3-VL adapter using Ollama's local chat API and JSON schema output."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")
        self.response_metadata: dict[str, Any] = {}

    @staticmethod
    def _valid_result(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        regions = result.get("regions")
        figures = result.get("figures")
        interpretation = result.get("page_interpretation")
        nested_regions = (
            interpretation.get("text_regions") if isinstance(interpretation, dict) else None
        )
        top_level_text_regions = result.get("text_regions")
        nested_figures = (
            interpretation.get("diagrams") if isinstance(interpretation, dict) else None
        )
        page_text = interpretation.get("text") if isinstance(interpretation, dict) else None
        page_diagrams = interpretation.get("diagram") if isinstance(interpretation, dict) else None
        page_drawings = interpretation.get("drawing") if isinstance(interpretation, dict) else None
        if (
            regions is None
            and figures is None
            and nested_regions is None
            and nested_figures is None
            and page_text is None
            and page_diagrams is None
            and page_drawings is None
            and top_level_text_regions is None
        ):
            return False
        if regions is not None:
            if not isinstance(regions, list):
                return False
            for region in regions:
                if not isinstance(region, dict):
                    return False
                if "bbox_2d" not in region and not all(
                    isinstance(region.get(key), (int, float))
                    for key in ("left", "top", "right", "bottom")
                ):
                    return False
        if figures is not None:
            if not isinstance(figures, list):
                return False
            for figure in figures:
                if not isinstance(figure, dict):
                    return False
                bbox = figure.get("bbox_2d")
                if not isinstance(figure.get("label"), str) or not (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(value, (int, float)) for value in bbox)
                ):
                    return False
        for nested in (nested_regions, nested_figures):
            if nested is not None and not (
                isinstance(nested, list)
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("bbox_2d"), list)
                    and len(item["bbox_2d"]) == 4
                    and all(isinstance(value, (int, float)) for value in item["bbox_2d"])
                    for item in nested
                )
            ):
                return False
        if top_level_text_regions is not None and not _valid_text_region_list(
            top_level_text_regions
        ):
            return False
        if nested_regions is not None and not _valid_text_region_list(nested_regions):
            return False
        if page_text is not None and not (
            isinstance(page_text, list) and all(isinstance(item, str) for item in page_text)
        ):
            return False
        if page_diagrams is not None:
            valid_diagrams = isinstance(page_diagrams, list) and all(
                isinstance(item, dict)
                and isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4
                and all(isinstance(value, (int, float)) for value in item["bbox"])
                and isinstance(item.get("description"), str)
                for item in page_diagrams
            )
            if not valid_diagrams:
                return False
        if page_drawings is not None:
            valid_drawings = isinstance(page_drawings, list) and all(
                isinstance(item, dict)
                and isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4
                and all(isinstance(value, (int, float)) for value in item["bbox"])
                and isinstance(item.get("description"), str)
                for item in page_drawings
            )
            if not valid_drawings:
                return False
        return True

    @staticmethod
    def _normalize_result(result: dict[str, Any], prepared_size: tuple[int, int]) -> dict[str, Any]:
        if "regions" in result:
            return result
        if "figures" in result:
            result["regions"] = [
                {
                    "order": index,
                    "kind": "figure",
                    "text": figure["label"],
                    "bbox_2d": figure["bbox_2d"],
                }
                for index, figure in enumerate(result["figures"])
            ]
            return result
        interpretation = result.get("page_interpretation", {})
        if not isinstance(interpretation, dict):
            return result
        page_text = interpretation.get("text", [])
        page_drawings = interpretation.get("drawing")
        if page_drawings is None:
            page_drawings = interpretation.get("diagram", [])
        prepared_width, prepared_height = prepared_size
        text_regions = cast(
            list[dict[str, Any]], result.get("text_regions", interpretation.get("text_regions", []))
        )
        diagrams = cast(list[dict[str, Any]], interpretation.get("diagrams", []))
        result["regions"] = [
            {
                "order": region.get("order", index),
                "kind": "text",
                "text": region.get("text"),
                "bbox_2d": region["bbox_2d"],
            }
            for index, region in enumerate(text_regions)
        ] + [
            {
                "order": len(text_regions) + index,
                "kind": "figure",
                "text": diagram.get("description"),
                "bbox_2d": diagram["bbox_2d"],
            }
            for index, diagram in enumerate(diagrams)
        ]
        result["regions"] += [
            {
                "order": len(result["regions"]) + index,
                "kind": "figure",
                "text": diagram.get("description"),
                "bbox_2d": [
                    round(diagram["bbox"][0] * prepared_width),
                    round(diagram["bbox"][1] * prepared_height),
                    round(diagram["bbox"][2] * prepared_width),
                    round(diagram["bbox"][3] * prepared_height),
                ],
            }
            for index, diagram in enumerate(page_drawings or [])
        ]
        result["page_text"] = page_text
        return result

    @classmethod
    def _structured_message(
        cls, message: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
        """Parse complete JSON messages, never prose surrounding them."""
        saw_json = False
        unsupported_shape: str | None = None
        for field_name in ("content", "thinking"):
            candidate = message.get(field_name)
            if not isinstance(candidate, str):
                continue
            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            saw_json = True
            if cls._valid_result(result):
                return cast(dict[str, Any], result), saw_json, None
            if isinstance(result, dict):
                unsupported_shape = ", ".join(sorted(result)) or "object"
        return None, saw_json, unsupported_shape

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        schema = {
            "type": "object",
            "properties": {
                "page_interpretation": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "array", "items": {"type": "string"}},
                        "text_regions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order": {"type": "integer"},
                                    "kind": {"type": "string", "enum": ["text"]},
                                    "bbox_2d": {
                                        "type": "array",
                                        "items": {"type": "number", "minimum": 0, "maximum": 1000},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "text": {"type": "string"},
                                },
                                "required": ["order", "kind", "bbox_2d"],
                            },
                        },
                        "drawings": {"type": "array"},
                    },
                    "required": ["text"],
                },
            },
            "required": ["page_interpretation"],
        }
        prompt = (
            "Transform this handwritten page. Return only JSON matching the schema. "
            "Transcribe the ordinary handwritten notes and textual marks visible on the page "
            "in reading "
            "order. Read the actual handwriting and preserve the wording, spelling, "
            "capitalization, punctuation, shorthand, symbols that are genuinely part "
            "of written text, and unfamiliar terminology. Do not autocorrect, replace "
            "a word with a semantically more likely word, normalize unfamiliar words, "
            "or invent text. Use [unclear] only when the letterforms are genuinely "
            "unreadable. Do not transcribe graphical elements of drawings or diagrams "
            "as page text, including arrows, diagram strokes, lines, and graphical "
            "connectors that clearly function as graphics. A handwritten word remains "
            "text even when it is physically near a drawing; do not exclude text merely "
            "because it is beside, above, below, or adjacent to a figure. Do not "
            "Do not treat arrows, diagram strokes, connectors, grafting cuts, or isolated "
            "drawing symbols as text. Return page-level text plus spatial text regions "
            "(`text_regions`) in "
            "reading order. Each text region must be a coherent handwritten block with "
            "order, kind=text, bbox_2d in Qwen3-VL 0..1000 coordinates, and optional text "
            "as model evidence. Do not make one region per word unless it is independently "
            "identified. Keep drawing content out of text_regions; the dedicated drawing "
            "localization pass owns figures. Qwen region text is evidence, not the canonical "
            "transcription."
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
        result, saw_json, unsupported_shape = self._structured_message(
            message if isinstance(message, dict) else {}
        )
        self.response_metadata = {
            "prompt_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
            "response_fields": [
                field_name
                for field_name in ("content", "thinking")
                if isinstance(message, dict) and isinstance(message.get(field_name), str)
            ],
        }
        if result is None:
            if saw_json:
                structured_error = (
                    "Qwen returned valid JSON but unsupported transform schema: "
                    f"{unsupported_shape or 'unknown'}"
                )
            else:
                structured_error = (
                    "Ollama/Qwen returned no valid structured JSON in content or thinking"
                )
            return {
                "status": "failure",
                "error": structured_error,
                "raw_response": body,
            }, (time.perf_counter() - started) * 1000
        return self._normalize_result(result, image.size), (time.perf_counter() - started) * 1000

    def release(self) -> None:
        # keep_alive=0 on the interpretation request asks Ollama to release the model.
        return None


def _valid_qwen_bbox(bbox: Any) -> bool:
    """Validate Qwen3-VL's documented 0-1000 drawing coordinates."""
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in bbox
        )
    ):
        return False
    return (
        all(0 <= value <= 1000 for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )


class OllamaDrawingLocalizer:
    """Dedicated Qwen3-VL boundary for semantic drawing localization."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")
        self.response_metadata: dict[str, Any] = {}

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> tuple[dict[str, Any] | None, str | None, list[Any]]:
        if not isinstance(result, dict):
            return None, "expected a JSON object", []
        drawings = result.get("drawings")
        if not isinstance(drawings, list):
            return None, "expected top-level drawings to be an array", []
        normalized: list[dict[str, Any]] = []
        for index, drawing in enumerate(drawings):
            if not isinstance(drawing, dict):
                return (
                    None,
                    f"drawing entry {index} has unsupported shape: expected an object",
                    [drawing],
                )
            bbox_key = "bbox_2d" if "bbox_2d" in drawing else "bbox"
            bbox = drawing.get(bbox_key)
            if not _valid_qwen_bbox(bbox):
                return (
                    None,
                    (
                        f"drawing entry {index} has invalid bbox: expected Qwen3-VL "
                        "0-1000 bbox_2d or bbox [x1, y1, x2, y2]"
                    ),
                    [drawing],
                )
            description = drawing.get("description")
            if description is not None and not isinstance(description, str):
                return None, f"drawing entry {index} has unsupported description type", [drawing]
            bbox_values = tuple(float(value) for value in cast(list[Any], bbox))
            normalized_drawing: dict[str, Any] = {
                "bbox_2d": list(bbox_values),
                "model_bbox": list(bbox_values),
                "bbox_coordinate_space": "qwen_0_1000",
            }
            if description is not None:
                normalized_drawing["description"] = description
            normalized.append(normalized_drawing)
        return {"drawings": normalized}, None, []

    @classmethod
    def _structured_message(
        cls, message: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, bool, str | None, list[Any]]:
        saw_json = False
        unsupported_shape: str | None = None
        unsupported_entries: list[Any] = []
        for field_name in ("content", "thinking"):
            candidate = message.get(field_name)
            if not isinstance(candidate, str):
                continue
            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            saw_json = True
            normalized, reason, entries = cls._normalize_result(result)
            if normalized is not None:
                return normalized, saw_json, None, []
            unsupported_shape = reason or "unsupported JSON value"
            unsupported_entries = entries
        return None, saw_json, unsupported_shape, unsupported_entries

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        schema = {
            "type": "object",
            "properties": {
                "drawings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0, "maximum": 1},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["bbox_2d"],
                    },
                }
            },
            "required": ["drawings"],
        }
        prompt = (
            "Identify every distinct hand-drawn diagram, illustration, sketch, or visual "
            "figure on this handwritten notebook page. Do not include ordinary handwriting, "
            "headings, bullets, isolated words, page decorations, notebook dots/grid, or "
            "individual letters that are part of ordinary text. A figure may contain "
            "disconnected strokes, arrows, labels, whitespace, and multiple sequential stages; "
            "treat a complete visual sequence as one figure when its parts clearly belong "
            "together. For each figure, return one approximate normalized [x1, y1, x2, y2] "
            "box covering the complete figure, including disconnected strokes, arrows, labels, "
            "annotations, and relationship-preserving whitespace. Do not transcribe, OCR, or "
            "enumerate text blocks. Return only the requested structured JSON."
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
            self.response_metadata = {"response_fields": []}
            return {
                "status": "failure",
                "error": (
                    f"Unable to query Ollama/Qwen drawing localization ({self.model}): {error}"
                ),
            }, (time.perf_counter() - started) * 1000
        message = body.get("message", {})
        result, saw_json, unsupported_shape, unsupported_entries = self._structured_message(
            message if isinstance(message, dict) else {}
        )
        self.response_metadata = {
            "prompt_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
            "response_fields": [
                field_name
                for field_name in ("content", "thinking")
                if isinstance(message, dict) and isinstance(message.get(field_name), str)
            ],
        }
        if result is None:
            if saw_json:
                structured_error = (
                    "Qwen returned valid drawing localization JSON but "
                    f"unsupported structure: {unsupported_shape or 'unknown'}"
                )
                if unsupported_entries:
                    structured_error += "; actual unsupported entry JSON: " + json.dumps(
                        unsupported_entries[0], sort_keys=True
                    )
            else:
                structured_error = (
                    "Ollama/Qwen returned no valid drawing localization JSON in content or thinking"
                )
            return {
                "status": "failure",
                "error": structured_error,
                "unsupported_entries": unsupported_entries,
                "raw_response": body,
            }, (time.perf_counter() - started) * 1000
        return result, (time.perf_counter() - started) * 1000

    def release(self) -> None:
        return None


def pad_normalized_bounds(
    bounds: tuple[float, float, float, float], proportion: float = 0.05
) -> tuple[float, float, float, float]:
    """Add deterministic proportional containment padding without shrinking a box."""
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    horizontal_padding = width * proportion
    vertical_padding = height * proportion
    return (
        max(0.0, left - horizontal_padding),
        max(0.0, top - vertical_padding),
        min(1.0, right + horizontal_padding),
        min(1.0, bottom + vertical_padding),
    )


def qwen_bbox_to_normalized(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert Qwen3-VL 0-1000 coordinates to the internal 0..1 form."""
    return cast(
        tuple[float, float, float, float],
        tuple(value / 1000 for value in bounds),
    )


def map_prepared_bounds(
    bounds: tuple[float, float, float, float],
    prepared_size: tuple[int, int],
    source_size: tuple[int, int],
) -> RegionBounds:
    """Map prepared-image pixel coordinates to clamped source-image pixels."""
    prepared_width, prepared_height = prepared_size
    source_width, source_height = source_size
    left, top, right, bottom = bounds
    mapped = (
        round(left * source_width / prepared_width),
        round(top * source_height / prepared_height),
        round(right * source_width / prepared_width),
        round(bottom * source_height / prepared_height),
    )
    left, top, right, bottom = mapped
    left, right = sorted((max(0, min(source_width, left)), max(0, min(source_width, right))))
    top, bottom = sorted((max(0, min(source_height, top)), max(0, min(source_height, bottom))))
    if right <= left or bottom <= top:
        raise ValueError("Qwen returned an empty region")
    return RegionBounds(left, top, right, bottom)


def _bounds(
    item: dict[str, Any],
    source_size: tuple[int, int],
    prepared_size: tuple[int, int],
) -> tuple[RegionBounds, RegionBounds | None]:
    width, height = source_size
    bbox = item.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        prepared_bounds = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
        return map_prepared_bounds(prepared_bounds, prepared_size, source_size), RegionBounds(
            *(round(value) for value in prepared_bounds)
        )
    values = [float(item.get(key, 0)) for key in ("left", "top", "right", "bottom")]
    normalized = RegionBounds(
        *(
            round(value * dimension)
            for value, dimension in zip(values, (width, height, width, height), strict=True)
        )
    )
    return map_prepared_bounds(
        (
            normalized.left * prepared_size[0] / width,
            normalized.top * prepared_size[1] / height,
            normalized.right * prepared_size[0] / width,
            normalized.bottom * prepared_size[1] / height,
        ),
        prepared_size,
        source_size,
    ), None


def _text_region_bounds(
    item: dict[str, Any], source_size: tuple[int, int], prepared_size: tuple[int, int]
) -> tuple[RegionBounds, RegionBounds, tuple[float, float, float, float]]:
    """Map a text region from Qwen's 0..1000 space to an original crop."""
    bbox = item.get("bbox_2d")
    if not _valid_qwen_bbox(bbox):
        raise ValueError("spatial text region has an invalid Qwen 0-1000 bbox")
    qwen_values = cast(list[float], bbox)
    qwen_bbox = cast(
        tuple[float, float, float, float], tuple(float(value) for value in qwen_values)
    )
    normalized = qwen_bbox_to_normalized(qwen_bbox)
    padded = pad_normalized_bounds(normalized, TEXT_REGION_PADDING)
    prepared_bounds = _normalized_to_prepared_bounds(padded, prepared_size)
    source_bounds = map_prepared_bounds(
        (
            prepared_bounds.left,
            prepared_bounds.top,
            prepared_bounds.right,
            prepared_bounds.bottom,
        ),
        prepared_size,
        source_size,
    )
    return source_bounds, prepared_bounds, normalized


def _normalized_to_prepared_bounds(
    bounds: tuple[float, float, float, float], prepared_size: tuple[int, int]
) -> RegionBounds:
    width, height = prepared_size
    left, top, right, bottom = (
        round(bounds[0] * width),
        round(bounds[1] * height),
        round(bounds[2] * width),
        round(bounds[3] * height),
    )
    return RegionBounds(
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(1, min(width, right)),
        max(1, min(height, bottom)),
    )


def _legacy_drawing_normalized(
    raw: dict[str, Any], source_size: tuple[int, int], prepared_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    _, prepared_bounds = _bounds(raw, source_size, prepared_size)
    if prepared_bounds is None:
        source_bounds, _ = _bounds(raw, source_size, prepared_size)
        return (
            source_bounds.left / source_size[0],
            source_bounds.top / source_size[1],
            source_bounds.right / source_size[0],
            source_bounds.bottom / source_size[1],
        )
    return (
        prepared_bounds.left / prepared_size[0],
        prepared_bounds.top / prepared_size[1],
        prepared_bounds.right / prepared_size[0],
        prepared_bounds.bottom / prepared_size[1],
    )


def transform_page(
    image_path: Path,
    interpreter: PageInterpreter | None = None,
    recognizer: Recognizer | None = None,
    *,
    recognizer_metadata: dict[str, Any] | None = None,
    drawing_localizer: DrawingLocalizer | None = None,
) -> TransformedPage:
    """Transform page text, then independently localize and crop its drawings."""
    transform_started = time.perf_counter()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error

    prepared = prepare_vlm_image_with_metadata(source)
    supplied_interpreter = interpreter is not None
    interpreter = interpreter or OllamaPageInterpreter()
    interpretation, page_transform_ms = interpreter.interpret(prepared.image)
    interpreter.release()
    if interpretation.get("status") == "failure":
        failure_error = interpretation.get("error", "Qwen interpretation failed")
        raise RuntimeError(f"{failure_error}: {json.dumps(interpretation.get('raw_response'))}")
    raw_regions = sorted(
        interpretation.get("regions", []), key=lambda item: int(item.get("order", 0))
    )
    spatial_text_raw = [raw for raw in raw_regions if str(raw.get("kind", "text")) != "figure"]
    if drawing_localizer is None and not supplied_interpreter:
        drawing_localizer = OllamaDrawingLocalizer()

    drawing_localization_ms = 0.0
    drawing_localization_runtime: dict[str, Any] = {
        "status": "not_run",
        "model": getattr(drawing_localizer, "model", None),
    }
    drawing_entries: list[
        tuple[int, tuple[float, float, float, float], str, str | None, dict[str, Any]]
    ]
    if drawing_localizer is not None:
        try:
            localized, drawing_localization_ms = drawing_localizer.localize(prepared.image)
            drawing_localizer.release()
        except (RuntimeError, ValueError) as error:
            localized = {
                "status": "failure",
                "error": str(error),
            }
            drawing_localization_runtime["status"] = "failure"
            drawing_localization_runtime["error"] = str(error)
        if localized.get("status") == "failure":
            drawing_localization_runtime.update(
                {
                    "status": "failure",
                    "error": localized.get("error", "drawing localization failed"),
                    "unsupported_entries": localized.get("unsupported_entries", []),
                }
            )
            drawing_entries = []
        else:
            drawing_localization_runtime["status"] = "success"
            drawing_entries = []
            for index, drawing in enumerate(localized.get("drawings", [])):
                model_bbox = cast(
                    tuple[float, float, float, float],
                    tuple(
                        float(value)
                        for value in drawing.get("model_bbox", drawing["bbox_2d"])
                    ),
                )
                drawing_entries.append(
                    (
                        index,
                        model_bbox,
                        str(drawing.get("bbox_coordinate_space", "normalized")),
                        cast(str | None, drawing.get("description")),
                        cast(dict[str, Any], drawing),
                    )
                )
    else:
        # Compatibility for existing mocked page responses. The production path
        # always uses the dedicated localizer above.
        drawing_entries = [
            (
                int(raw.get("order", index)),
                _legacy_drawing_normalized(raw, source.size, prepared.prepared_dimensions),
                "normalized",
                raw.get("text"),
                raw,
            )
            for index, raw in enumerate(raw_regions)
            if str(raw.get("kind", "text")) == "figure"
        ]
        if drawing_entries:
            drawing_localization_runtime["status"] = "legacy_page_response"
    if spatial_text_raw and recognizer is None:
        recognizer, load_ms, cuda, device, gpu = TrocrRecognizer.from_local_cache()
        recognizer_metadata = {
            "model_load_ms": load_ms,
            "cuda_available": cuda,
            "device": device,
            "gpu": gpu,
        }
    elif not spatial_text_raw:
        recognizer_metadata = recognizer_metadata or {
            "status": "not_applicable",
            "reason": "no spatial text regions",
        }
    else:
        recognizer_metadata = recognizer_metadata or {}
    regions: list[TransformedRegion] = []
    artifact_directory = image_path.parent / f"{image_path.stem}.sibyl" / "assets"
    figure_count = 0
    disagreements: list[dict[str, Any]] = []
    trocr_timings: list[dict[str, Any]] = []
    trocr_successes = 0
    trocr_failures = 0
    crop_started = time.perf_counter()
    for text_index, raw in enumerate(spatial_text_raw, start=1):
        if isinstance(raw.get("bbox_2d"), list):
            bounds, prepared_bounds, normalized_bounds = _text_region_bounds(
                raw, source.size, prepared.prepared_dimensions
            )
        else:
            # Compatibility for older mocked page responses using normalized edge keys.
            bounds, legacy_prepared_bounds = _bounds(
                raw, source.size, prepared.prepared_dimensions
            )
            normalized_bounds = (
                bounds.left / source.width,
                bounds.top / source.height,
                bounds.right / source.width,
                bounds.bottom / source.height,
            )
            prepared_bounds = legacy_prepared_bounds or _normalized_to_prepared_bounds(
                normalized_bounds, prepared.prepared_dimensions
            )
        region_image = source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert(
            "RGB"
        )
        artifact_directory.mkdir(parents=True, exist_ok=True)
        text_asset_name = f"text-{text_index:02d}.png"
        text_asset_path = artifact_directory / text_asset_name
        region_image.save(text_asset_path, format="PNG")
        kind = str(raw.get("kind", "text"))
        if recognizer is None:
            raise RuntimeError("TrOCR recognizer is unavailable for a spatial text region")
        started = time.perf_counter()
        order = int(raw.get("order", len(regions)))
        try:
            text, inference_ms = recognizer.recognize(region_image)
            elapsed_ms = (time.perf_counter() - started) * 1000
            recognizer_observation = {
                "status": "success",
                "text": text,
                "inference_ms": round(inference_ms, 3),
                "elapsed_ms": round(elapsed_ms, 3),
            }
            trocr_successes += 1
            trocr_timings.append({"order": order, "inference_ms": round(inference_ms, 3)})
        except Exception as error:  # A bad crop must not discard the page transform.
            elapsed_ms = (time.perf_counter() - started) * 1000
            text = "[unclear]"
            inference_ms = 0.0
            trocr_failures += 1
            recognizer_observation = {
                "status": "failure",
                "error": str(error),
                "inference_ms": None,
                "elapsed_ms": round(elapsed_ms, 3),
            }
            trocr_timings.append(
                {"order": order, "status": "failure", "elapsed_ms": round(elapsed_ms, 3)}
            )
        if raw.get("text") is not None and raw.get("text") != text:
            disagreements.append(
                {
                    "order": order,
                    "qwen": raw.get("text"),
                    "trocr": text,
                    "status": recognizer_observation["status"],
                }
            )
        source_metadata: dict[str, Any] = {
            "image": str(image_path),
            "bounds": asdict(bounds),
            "prepared_bounds": asdict(prepared_bounds) if prepared_bounds else None,
            "model_bbox": raw.get("bbox_2d"),
            "bbox_coordinate_space": "qwen_0_1000" if raw.get("bbox_2d") else "legacy_normalized",
            "normalized_bounds": normalized_bounds,
            "text_crop": f"assets/{text_asset_name}",
            "crop": {
                "source_bounds": [bounds.left, bounds.top, bounds.right, bounds.bottom],
                "prepared_bounds": (
                    [
                        prepared_bounds.left,
                        prepared_bounds.top,
                        prepared_bounds.right,
                        prepared_bounds.bottom,
                    ]
                    if prepared_bounds
                    else None
                ),
                "width": region_image.width,
                "height": region_image.height,
                "representation": "RGB source crop passed to TrOCR",
            },
            "padding": {
                "normalized_proportion": TEXT_REGION_PADDING
                if raw.get("bbox_2d")
                else 0.0,
                "normalized_bounds": pad_normalized_bounds(
                    normalized_bounds,
                    TEXT_REGION_PADDING if raw.get("bbox_2d") else 0.0,
                ),
            },
            "provenance": ["page_text_localization", "trocr"],
        }
        regions.append(
            TransformedRegion(
                order=order,
                kind=kind,
                bounds=bounds,
                prepared_bounds=prepared_bounds,
                qwen_text=raw.get("text"),
                text=text,
                normalized_bounds=normalized_bounds,
                source=source_metadata,
                recognizer=recognizer_observation,
            )
        )
    for (
        order,
        model_bbox,
        coordinate_space,
        description,
        localization_observation,
    ) in drawing_entries:
        if coordinate_space == "qwen_0_1000":
            normalized_bounds = qwen_bbox_to_normalized(model_bbox)
        elif coordinate_space == "normalized":
            # Compatibility for page-response fixtures that already provide 0..1 values.
            normalized_bounds = model_bbox
        else:
            raise ValueError(f"unsupported drawing coordinate space: {coordinate_space}")
        padded_normalized = (
            normalized_bounds
            if drawing_localization_runtime["status"] == "legacy_page_response"
            else pad_normalized_bounds(normalized_bounds)
        )
        prepared_bounds = _normalized_to_prepared_bounds(
            padded_normalized, prepared.prepared_dimensions
        )
        bounds = map_prepared_bounds(
            (
                prepared_bounds.left,
                prepared_bounds.top,
                prepared_bounds.right,
                prepared_bounds.bottom,
            ),
            prepared.prepared_dimensions,
            source.size,
        )
        artifact_directory.mkdir(parents=True, exist_ok=True)
        figure_count += 1
        crop_path = artifact_directory / f"figure-{figure_count:02d}.png"
        source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert("RGB").save(
            crop_path, format="PNG"
        )
        source_metadata = {
            "image": str(image_path),
            "bounds": asdict(bounds),
            "prepared_bounds": asdict(prepared_bounds),
            "model_bbox": list(model_bbox),
            "bbox_coordinate_space": coordinate_space,
            "normalized_bounds": normalized_bounds,
            "padded_normalized_bounds": padded_normalized,
            "crop": str(crop_path),
            "provenance": ["drawing_localization"],
            "drawing_localization": localization_observation,
        }
        regions.append(
            TransformedRegion(
                order=order,
                kind="figure",
                bounds=bounds,
                prepared_bounds=prepared_bounds,
                qwen_text=description,
                text="",
                normalized_bounds=normalized_bounds,
                source=source_metadata,
                recognizer={"status": "not_applicable"},
            )
        )
    crop_ms = (
        (time.perf_counter() - crop_started) * 1000
        if (figure_count or spatial_text_raw)
        else 0.0
    )
    response_metadata = getattr(interpreter, "response_metadata", {})
    total_ms = (time.perf_counter() - transform_started) * 1000
    drawing_count = sum(region.kind == "figure" for region in regions)
    spatial_text_count = len(regions) - drawing_count
    page_text = (
        [
            region.text
            for region in sorted(regions, key=lambda item: item.order)
            if region.kind != "figure"
        ]
        if spatial_text_raw
        else interpretation.get(
            "page_text", interpretation.get("page_interpretation", {}).get("text", [])
        )
    )
    runtime = {
        "vlm_model": getattr(interpreter, "model", None),
        "vlm_ms": round(page_transform_ms, 3),
        "vlm_dimensions": {
            "width": prepared.prepared_dimensions[0],
            "height": prepared.prepared_dimensions[1],
        },
        "page_transform": {
            "status": "success",
            "model": getattr(interpreter, "model", None),
            "timing_ms": round(page_transform_ms, 3),
            "response_metadata": response_metadata,
        },
        "text_localization": {
            "status": "success",
            "timing_ms": round(page_transform_ms, 3),
            "regions": spatial_text_count,
        },
        "drawing_localization": {
            **drawing_localization_runtime,
            "timing_ms": round(drawing_localization_ms, 3),
            "response_metadata": getattr(drawing_localizer, "response_metadata", {}),
        },
        "recognizer": recognizer_metadata,
        "benchmark": {
            "model": getattr(interpreter, "model", None),
            "page_transform_model": getattr(interpreter, "model", None),
            "drawing_localization_model": getattr(drawing_localizer, "model", None),
            "runtime": "ollama",
            "preparation_dimensions": {
                "width": prepared.prepared_dimensions[0],
                "height": prepared.prepared_dimensions[1],
            },
            "source_dimensions": {"width": source.width, "height": source.height},
            "scale": prepared.scale,
            "preparation_ms": round(prepared.preparation_ms, 3),
            "qwen_ms": round(page_transform_ms, 3),
            "page_transform_ms": round(page_transform_ms, 3),
            "drawing_localization_ms": round(drawing_localization_ms, 3),
            "text_localization_ms": round(page_transform_ms, 3),
            "crop_ms": round(crop_ms, 3),
            "prompt_tokens": response_metadata.get("prompt_tokens"),
            "output_tokens": response_metadata.get("output_tokens"),
            "region_count": len(regions),
            "spatial_text_regions": spatial_text_count,
            "drawing_regions": drawing_count,
            "trocr_attempts": len(trocr_timings),
            "trocr_successes": trocr_successes,
            "trocr_failures": trocr_failures,
            "trocr_ms": round(
                sum(float(timing.get("inference_ms") or 0) for timing in trocr_timings), 3
            ),
            "trocr_timings": trocr_timings,
            "total_transform_ms": round(total_ms, 3),
        },
        "disagreements": disagreements,
        "canonical_text_source": "trocr_spatial_regions" if spatial_text_raw else "qwen_page_text",
        "qwen_page_text": interpretation.get(
            "page_text", interpretation.get("page_interpretation", {}).get("text", [])
        ),
    }
    return TransformedPage(
        source={"image": str(image_path)},
        dimensions={"width": source.width, "height": source.height},
        interpretation=interpretation.get("page_interpretation", {}),
        regions=regions,
        runtime=runtime,
        page_text=page_text,
    )


def format_transform(page: TransformedPage) -> str:
    return json.dumps(asdict(page), indent=2)


def write_transform_json(page: TransformedPage) -> Path:
    """Persist the complete structured transform beside its projections."""
    source_path = Path(page.source["image"])
    output_directory = source_path.parent / f"{source_path.stem}.sibyl"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "transform.json"
    output_path.write_text(f"{format_transform(page)}\n", encoding="utf-8")
    return output_path


def format_text_transform(page: TransformedPage) -> str:
    """Project only transformed text, preserving model spelling and order."""
    lines: list[str] = list(page.page_text)
    assembled_spatial_text = page.runtime.get("canonical_text_source") == "trocr_spatial_regions"
    for region in sorted(page.regions, key=lambda item: item.order):
        if region.kind == "figure" or not region.text or assembled_spatial_text:
            continue
        if region.kind in {"bullet", "list_item"}:
            lines.append(f"- {region.text}")
        else:
            lines.append(region.text)
    return "\n\n".join(lines)


def write_markdown_transform(page: TransformedPage) -> Path:
    """Write the Markdown projection beside its original-resolution assets."""
    source_path = Path(page.source["image"])
    output_directory = source_path.parent / f"{source_path.stem}.sibyl"
    assets_directory = output_directory / "assets"
    output_directory.mkdir(parents=True, exist_ok=True)
    write_transform_json(page)
    markdown_lines: list[str] = list(page.page_text)
    assembled_spatial_text = page.runtime.get("canonical_text_source") == "trocr_spatial_regions"
    figure_count = 0
    for region in sorted(page.regions, key=lambda item: item.order):
        if region.kind == "figure":
            figure_count += 1
            crop_value = region.source.get("crop")
            if crop_value is None:
                continue
            crop_path = Path(crop_value)
            target = assets_directory / f"figure-{figure_count:02d}.png"
            if crop_path.resolve() != target.resolve():
                assets_directory.mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_path, target)
            label = f"Figure {figure_count}"
            markdown_lines.append(f"![{label}](assets/{target.name})")
            continue
        if not region.text or assembled_spatial_text:
            continue
        if region.kind == "heading":
            markdown_lines.append(f"# {region.text}")
        elif region.kind in {"bullet", "list_item"}:
            markdown_lines.append(f"- {region.text}")
        else:
            markdown_lines.append(region.text)
    output_path = output_directory / "transform.md"
    output_path.write_text("\n\n".join(markdown_lines) + "\n", encoding="utf-8")
    return output_path

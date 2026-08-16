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
from statistics import median
from typing import Any, Protocol, cast

from PIL import Image, ImageChops, UnidentifiedImageError

DRAWING_MAX_DIMENSIONS = (1536, 2048)
CONTENT_PAGE_MAX_DIMENSIONS = (1536, 2048)
DEFAULT_PAGE_MAX_DIMENSION = 1536
DEFAULT_PAGE_FOCUS = "full"
CONTENT_PAGE_FOCUS = "content"
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
    focus: str = DEFAULT_PAGE_FOCUS


@dataclass(frozen=True)
class TransformedRegion:
    order: int
    kind: str
    bounds: RegionBounds
    prepared_bounds: RegionBounds | None
    text: str
    normalized_bounds: tuple[float, float, float, float] | None = None
    source: dict[str, Any] = field(default_factory=dict)


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


def _bounded_dimensions(size: tuple[int, int], maximum: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    scale = min(maximum[0] / width, maximum[1] / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_vlm_image(source: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Create the fixed-size grayscale derivative used by drawing localization."""
    image = source.convert("L")
    dimensions = _bounded_dimensions(image.size, DRAWING_MAX_DIMENSIONS)
    if dimensions != image.size:
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
    return image, dimensions


def _page_max_dimension() -> int:
    configured = os.environ.get("SIBYL_PAGE_MAX_DIMENSION")
    if configured is None:
        return DEFAULT_PAGE_MAX_DIMENSION
    try:
        maximum = int(configured)
    except ValueError as error:
        raise ValueError("SIBYL_PAGE_MAX_DIMENSION must be a positive integer") from error
    if maximum <= 0:
        raise ValueError("SIBYL_PAGE_MAX_DIMENSION must be a positive integer")
    return maximum


def _page_focus() -> str:
    focus = os.environ.get("SIBYL_PAGE_FOCUS", DEFAULT_PAGE_FOCUS)
    if focus not in {DEFAULT_PAGE_FOCUS, CONTENT_PAGE_FOCUS}:
        raise ValueError("SIBYL_PAGE_FOCUS must be 'full' or 'content'")
    return focus


def _corner_background(image: Image.Image) -> tuple[int, int, int]:
    image = image.convert("RGB")
    patch_size = min(32, image.width, image.height)
    corners = (
        (0, 0, patch_size, patch_size),
        (image.width - patch_size, 0, image.width, patch_size),
        (0, image.height - patch_size, patch_size, image.height),
        (
            image.width - patch_size,
            image.height - patch_size,
            image.width,
            image.height,
        ),
    )
    pixels = [
        image.getpixel((x, y))
        for left, top, right, bottom in corners
        for y in range(top, bottom)
        for x in range(left, right)
    ]
    return cast(
        tuple[int, int, int],
        tuple(int(median(channel)) for channel in zip(*pixels, strict=True)),
    )


def _content_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    """Find non-background content without classifying or interpreting page marks."""
    rgb_image = image.convert("RGB")
    background = Image.new("RGB", rgb_image.size, _corner_background(rgb_image))
    difference = ImageChops.difference(rgb_image, background)
    foreground = difference.point(lambda value: 255 if value > 16 else 0)
    return foreground.getbbox() or (0, 0, rgb_image.width, rgb_image.height)


def prepare_page_image(source: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Create the page grayscale derivative with an experiment-controlled width maximum."""
    image = source.convert("L")
    focus = _page_focus()
    if focus == CONTENT_PAGE_FOCUS:
        image = image.crop(_content_bounds(image))
        dimensions = _bounded_dimensions(image.size, CONTENT_PAGE_MAX_DIMENSIONS)
    else:
        maximum = _page_max_dimension()
        height_maximum = round(maximum * image.height / image.width)
        dimensions = _bounded_dimensions(image.size, (maximum, height_maximum))
    if dimensions != image.size:
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
    return image, dimensions


def prepare_page_image_with_metadata(source: Image.Image) -> PreparedVlmImage:
    """Create the page inference representation and measure preparation."""
    started = time.perf_counter()
    focus = _page_focus()
    image, dimensions = prepare_page_image(source)
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
        focus=focus,
    )


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
        self.response_metadata: dict[str, Any] = {}

    @staticmethod
    def _valid_result(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        interpretation = result.get("page_interpretation")
        page_text = interpretation.get("text") if isinstance(interpretation, dict) else None
        return isinstance(page_text, list) and all(isinstance(item, str) for item in page_text)

    @staticmethod
    def _normalize_result(result: dict[str, Any], prepared_size: tuple[int, int]) -> dict[str, Any]:
        interpretation = result.get("page_interpretation", {})
        if not isinstance(interpretation, dict):
            return result
        page_text = interpretation.get("text", [])
        result["page_text"] = page_text
        result["regions"] = []
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
            "enumerate spatial text regions, drawings, or invent coordinates. Do not "
            "perform exhaustive text localization; a separate pass handles drawing "
            "localization."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"num_predict": 256},
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


def transform_page(
    image_path: Path,
    interpreter: PageInterpreter | None = None,
    *,
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

    page_prepared = prepare_page_image_with_metadata(source)
    drawing_image, drawing_dimensions = prepare_vlm_image(source)
    supplied_interpreter = interpreter is not None
    interpreter = interpreter or OllamaPageInterpreter()
    interpretation, page_transform_ms = interpreter.interpret(page_prepared.image)
    interpreter.release()
    if interpretation.get("status") == "failure":
        failure_error = interpretation.get("error", "Qwen interpretation failed")
        raise RuntimeError(f"{failure_error}: {json.dumps(interpretation.get('raw_response'))}")
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
            localized, drawing_localization_ms = drawing_localizer.localize(drawing_image)
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
        drawing_entries = []
    regions: list[TransformedRegion] = []
    artifact_directory = image_path.parent / f"{image_path.stem}.sibyl" / "assets"
    if artifact_directory.is_dir():
        stale_assets = [
            *artifact_directory.glob("text-*.png"),
            *artifact_directory.glob("figure-*.png"),
        ]
        for stale_asset in stale_assets:
            stale_asset.unlink()
    figure_count = 0
    crop_started = time.perf_counter()
    for (
        order,
        model_bbox,
        coordinate_space,
        _description,
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
            padded_normalized, drawing_dimensions
        )
        bounds = map_prepared_bounds(
            (
                prepared_bounds.left,
                prepared_bounds.top,
                prepared_bounds.right,
                prepared_bounds.bottom,
            ),
            drawing_dimensions,
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
                text="",
                normalized_bounds=normalized_bounds,
                source=source_metadata,
            )
        )
    crop_ms = (time.perf_counter() - crop_started) * 1000 if figure_count else 0.0
    response_metadata = getattr(interpreter, "response_metadata", {})
    total_ms = (time.perf_counter() - transform_started) * 1000
    drawing_count = sum(region.kind == "figure" for region in regions)
    page_text = interpretation.get(
        "page_text", interpretation.get("page_interpretation", {}).get("text", [])
    )
    runtime = {
        "vlm_model": getattr(interpreter, "model", None),
        "vlm_ms": round(page_transform_ms, 3),
        "vlm_dimensions": {
            "width": page_prepared.prepared_dimensions[0],
            "height": page_prepared.prepared_dimensions[1],
        },
        "page_transform": {
            "status": "success",
            "model": getattr(interpreter, "model", None),
            "page_focus": page_prepared.focus,
            "timing_ms": round(page_transform_ms, 3),
            "response_metadata": response_metadata,
        },
        "drawing_localization": {
            **drawing_localization_runtime,
            "timing_ms": round(drawing_localization_ms, 3),
            "response_metadata": getattr(drawing_localizer, "response_metadata", {}),
        },
        "benchmark": {
            "model": getattr(interpreter, "model", None),
            "page_transform_model": getattr(interpreter, "model", None),
            "drawing_localization_model": getattr(drawing_localizer, "model", None),
            "runtime": "ollama",
            "preparation_dimensions": {
                "width": page_prepared.prepared_dimensions[0],
                "height": page_prepared.prepared_dimensions[1],
            },
            "page_preparation_dimensions": {
                "width": page_prepared.prepared_dimensions[0],
                "height": page_prepared.prepared_dimensions[1],
            },
            "page_focus": page_prepared.focus,
            "drawing_preparation_dimensions": {
                "width": drawing_dimensions[0],
                "height": drawing_dimensions[1],
            },
            "source_dimensions": {"width": source.width, "height": source.height},
            "scale": page_prepared.scale,
            "preparation_ms": round(page_prepared.preparation_ms, 3),
            "qwen_ms": round(page_transform_ms, 3),
            "page_transform_ms": round(page_transform_ms, 3),
            "drawing_localization_ms": round(drawing_localization_ms, 3),
            "crop_ms": round(crop_ms, 3),
            "prompt_tokens": response_metadata.get("prompt_tokens"),
            "output_tokens": response_metadata.get("output_tokens"),
            "region_count": len(regions),
            "drawing_regions": drawing_count,
            "total_transform_ms": round(total_ms, 3),
        },
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
    return "\n\n".join(page.page_text)


def write_markdown_transform(page: TransformedPage) -> Path:
    """Write the Markdown projection beside its original-resolution assets."""
    source_path = Path(page.source["image"])
    output_directory = source_path.parent / f"{source_path.stem}.sibyl"
    assets_directory = output_directory / "assets"
    output_directory.mkdir(parents=True, exist_ok=True)
    write_transform_json(page)
    markdown_lines: list[str] = list(page.page_text)
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
    output_path = output_directory / "transform.md"
    output_path.write_text("\n\n".join(markdown_lines) + "\n", encoding="utf-8")
    return output_path

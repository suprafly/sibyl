"""One-page recovery orchestration and provider boundaries."""

from __future__ import annotations

import json
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
class RecoveredRegion:
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
class RecoveredPage:
    source: dict[str, Any]
    dimensions: dict[str, int]
    interpretation: dict[str, Any]
    regions: list[RecoveredRegion]
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
        text_regions = interpretation.get("text_regions", [])
        diagrams = interpretation.get("diagrams", [])
        result["regions"] = [
            {
                "order": index,
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
                    },
                    "required": ["text"],
                },
            },
            "required": ["page_interpretation"],
        }
        prompt = (
            "Recover this handwritten page. Return only JSON matching the schema. "
            "Return ordered page-level text exactly as observed. Do not enumerate "
            "spatial text regions, drawings, or invent coordinates. Do not perform "
            "exhaustive text localization; a separate pass handles drawing localization."
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
                    "Qwen returned valid JSON but unsupported recovery schema: "
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


class OllamaDrawingLocalizer:
    """Dedicated Qwen3-VL boundary for semantic drawing localization."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")
        self.response_metadata: dict[str, Any] = {}

    @staticmethod
    def _valid_result(result: Any) -> bool:
        if not isinstance(result, dict) or not isinstance(result.get("drawings"), list):
            return False
        for drawing in result["drawings"]:
            if not isinstance(drawing, dict):
                return False
            bbox = drawing.get("bbox_2d")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in bbox)
                and bbox[2] > bbox[0]
                and bbox[3] > bbox[1]
            ):
                return False
            if "description" in drawing and not isinstance(drawing["description"], str):
                return False
        return True

    @classmethod
    def _structured_message(
        cls, message: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
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
                    "Qwen returned valid JSON but unsupported drawing localization schema: "
                    f"{unsupported_shape or 'unknown'}"
                )
            else:
                structured_error = (
                    "Ollama/Qwen returned no valid drawing localization JSON in content or thinking"
                )
            return {
                "status": "failure",
                "error": structured_error,
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


def recover_page(
    image_path: Path,
    interpreter: PageInterpreter | None = None,
    recognizer: Recognizer | None = None,
    *,
    recognizer_metadata: dict[str, Any] | None = None,
    drawing_localizer: DrawingLocalizer | None = None,
) -> RecoveredPage:
    """Recover page text, then independently localize and crop its drawings."""
    recovery_started = time.perf_counter()
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
    interpretation, page_recovery_ms = interpreter.interpret(prepared.image)
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
    drawing_entries: list[tuple[int, tuple[float, float, float, float], str | None, dict[str, Any]]]
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
                }
            )
            drawing_entries = []
        else:
            drawing_localization_runtime["status"] = "success"
            drawing_entries = []
            for index, drawing in enumerate(localized.get("drawings", [])):
                normalized_bbox = cast(
                    tuple[float, float, float, float],
                    tuple(float(value) for value in drawing["bbox_2d"]),
                )
                drawing_entries.append(
                    (index, normalized_bbox, drawing.get("description"), drawing)
                )
    else:
        # Compatibility for existing mocked page responses. The production path
        # always uses the dedicated localizer above.
        drawing_entries = [
            (
                int(raw.get("order", index)),
                _legacy_drawing_normalized(raw, source.size, prepared.prepared_dimensions),
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
    regions: list[RecoveredRegion] = []
    artifact_directory = image_path.parent / f"{image_path.stem}.sibyl" / "assets"
    figure_count = 0
    disagreements: list[dict[str, Any]] = []
    trocr_timings: list[dict[str, Any]] = []
    for raw in spatial_text_raw:
        bounds, prepared_bounds = _bounds(raw, source.size, prepared.prepared_dimensions)
        region_image = source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert(
            "RGB"
        )
        kind = str(raw.get("kind", "text"))
        if recognizer is None:
            raise RuntimeError("TrOCR recognizer is unavailable for a spatial text region")
        started = time.perf_counter()
        text, inference_ms = recognizer.recognize(region_image)
        elapsed_ms = (time.perf_counter() - started) * 1000
        trocr_timings.append(
            {
                "order": int(raw.get("order", len(regions))),
                "inference_ms": round(inference_ms, 3),
            }
        )
        if raw.get("text") is not None and raw.get("text") != text:
            disagreements.append(
                {
                    "order": int(raw.get("order", len(regions))),
                    "qwen": raw.get("text"),
                    "trocr": text,
                }
            )
        source_metadata: dict[str, Any] = {"image": str(image_path), "bounds": asdict(bounds)}
        regions.append(
            RecoveredRegion(
                order=int(raw.get("order", len(regions))),
                kind=kind,
                bounds=bounds,
                prepared_bounds=prepared_bounds,
                qwen_text=raw.get("text"),
                text=text,
                normalized_bounds=None,
                source=source_metadata,
                recognizer={
                    "inference_ms": round(inference_ms, 3),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
            )
        )
    crop_started = time.perf_counter()
    for order, normalized_bounds, description, localization_observation in drawing_entries:
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
            "normalized_bounds": normalized_bounds,
            "padded_normalized_bounds": padded_normalized,
            "crop": str(crop_path),
            "provenance": ["drawing_localization"],
            "drawing_localization": localization_observation,
        }
        regions.append(
            RecoveredRegion(
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
    crop_ms = (time.perf_counter() - crop_started) * 1000 if figure_count else 0.0
    response_metadata = getattr(interpreter, "response_metadata", {})
    total_ms = (time.perf_counter() - recovery_started) * 1000
    drawing_count = sum(region.kind == "figure" for region in regions)
    spatial_text_count = len(regions) - drawing_count
    runtime = {
        "vlm_model": getattr(interpreter, "model", None),
        "vlm_ms": round(page_recovery_ms, 3),
        "vlm_dimensions": {
            "width": prepared.prepared_dimensions[0],
            "height": prepared.prepared_dimensions[1],
        },
        "page_recovery": {
            "status": "success",
            "model": getattr(interpreter, "model", None),
            "timing_ms": round(page_recovery_ms, 3),
            "response_metadata": response_metadata,
        },
        "drawing_localization": {
            **drawing_localization_runtime,
            "timing_ms": round(drawing_localization_ms, 3),
            "response_metadata": getattr(drawing_localizer, "response_metadata", {}),
        },
        "recognizer": recognizer_metadata,
        "benchmark": {
            "model": getattr(interpreter, "model", None),
            "page_recovery_model": getattr(interpreter, "model", None),
            "drawing_localization_model": getattr(drawing_localizer, "model", None),
            "runtime": "ollama",
            "preparation_dimensions": {
                "width": prepared.prepared_dimensions[0],
                "height": prepared.prepared_dimensions[1],
            },
            "source_dimensions": {"width": source.width, "height": source.height},
            "scale": prepared.scale,
            "preparation_ms": round(prepared.preparation_ms, 3),
            "qwen_ms": round(page_recovery_ms, 3),
            "page_recovery_ms": round(page_recovery_ms, 3),
            "drawing_localization_ms": round(drawing_localization_ms, 3),
            "crop_ms": round(crop_ms, 3),
            "prompt_tokens": response_metadata.get("prompt_tokens"),
            "output_tokens": response_metadata.get("output_tokens"),
            "region_count": len(regions),
            "spatial_text_regions": spatial_text_count,
            "drawing_regions": drawing_count,
            "trocr_attempts": len(trocr_timings),
            "trocr_timings": trocr_timings,
            "total_recovery_ms": round(total_ms, 3),
        },
        "disagreements": disagreements,
    }
    return RecoveredPage(
        source={"image": str(image_path)},
        dimensions={"width": source.width, "height": source.height},
        interpretation=interpretation.get("page_interpretation", {}),
        regions=regions,
        runtime=runtime,
        page_text=interpretation.get(
            "page_text", interpretation.get("page_interpretation", {}).get("text", [])
        ),
    )


def format_recovery(page: RecoveredPage) -> str:
    return json.dumps(asdict(page), indent=2)


def write_recovery_json(page: RecoveredPage) -> Path:
    """Persist the complete structured recovery beside its projections."""
    source_path = Path(page.source["image"])
    output_directory = source_path.parent / f"{source_path.stem}.sibyl"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "recovery.json"
    output_path.write_text(f"{format_recovery(page)}\n", encoding="utf-8")
    return output_path


def format_text_recovery(page: RecoveredPage) -> str:
    """Project only recovered text, preserving model spelling and order."""
    lines: list[str] = list(page.page_text)
    for region in sorted(page.regions, key=lambda item: item.order):
        if region.kind == "figure" or not region.text:
            continue
        if region.kind in {"bullet", "list_item"}:
            lines.append(f"- {region.text}")
        else:
            lines.append(region.text)
    return "\n\n".join(lines)


def write_markdown_recovery(page: RecoveredPage) -> Path:
    """Write the Markdown projection beside its original-resolution assets."""
    source_path = Path(page.source["image"])
    output_directory = source_path.parent / f"{source_path.stem}.sibyl"
    assets_directory = output_directory / "assets"
    output_directory.mkdir(parents=True, exist_ok=True)
    write_recovery_json(page)
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
        if not region.text:
            continue
        if region.kind == "heading":
            markdown_lines.append(f"# {region.text}")
        elif region.kind in {"bullet", "list_item"}:
            markdown_lines.append(f"- {region.text}")
        else:
            markdown_lines.append(region.text)
    output_path = output_directory / "recovery.md"
    output_path.write_text("\n\n".join(markdown_lines) + "\n", encoding="utf-8")
    return output_path

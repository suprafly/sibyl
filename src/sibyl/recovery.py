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
        if (
            regions is None
            and figures is None
            and nested_regions is None
            and nested_figures is None
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
        return True

    @staticmethod
    def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
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
        return result

    @classmethod
    def _structured_message(cls, message: dict[str, Any]) -> dict[str, Any] | None:
        """Parse only a complete JSON message, never prose surrounding it."""
        for field_name in ("content", "thinking"):
            candidate = message.get(field_name)
            if not isinstance(candidate, str):
                continue
            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if cls._valid_result(result):
                return cast(dict[str, Any], result)
        return None

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
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["kind", "order"],
                    },
                },
                "figures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["label", "bbox_2d"],
                    },
                },
            },
            "required": [],
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
        result = self._structured_message(message if isinstance(message, dict) else {})
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
            return {
                "status": "failure",
                "error": "Ollama/Qwen returned no valid structured JSON in content or thinking",
                "raw_response": body,
            }, (time.perf_counter() - started) * 1000
        return self._normalize_result(result), (time.perf_counter() - started) * 1000

    def release(self) -> None:
        # keep_alive=0 on the interpretation request asks Ollama to release the model.
        return None


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
        *(round(value * dimension) for value, dimension in zip(
            values, (width, height, width, height), strict=True
        ))
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


def recover_page(
    image_path: Path,
    interpreter: PageInterpreter | None = None,
    recognizer: Recognizer | None = None,
    *,
    recognizer_metadata: dict[str, Any] | None = None,
) -> RecoveredPage:
    """Interpret one source page, then recognize its identified regions."""
    recovery_started = time.perf_counter()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error

    prepared = prepare_vlm_image_with_metadata(source)
    interpreter = interpreter or OllamaPageInterpreter()
    interpretation, qwen_ms = interpreter.interpret(prepared.image)
    interpreter.release()
    if interpretation.get("status") == "failure":
        failure_error = interpretation.get("error", "Qwen interpretation failed")
        raise RuntimeError(f"{failure_error}: {json.dumps(interpretation.get('raw_response'))}")
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
    artifact_directory = image_path.parent / f"{image_path.stem}.sibyl" / "assets"
    figure_count = 0
    disagreements: list[dict[str, Any]] = []
    trocr_timings: list[dict[str, Any]] = []
    for raw in sorted(interpretation["regions"], key=lambda item: int(item.get("order", 0))):
        bounds, prepared_bounds = _bounds(raw, source.size, prepared.prepared_dimensions)
        region_image = source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert(
            "RGB"
        )
        kind = str(raw.get("kind", "text"))
        crop_path: Path | None = None
        if kind == "figure":
            figure_count += 1
            artifact_directory.mkdir(parents=True, exist_ok=True)
            crop_path = artifact_directory / f"figure-{figure_count:02d}.png"
            region_image.save(crop_path, format="PNG")
            text = ""
            inference_ms = 0.0
            elapsed_ms = 0.0
        else:
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
        if crop_path is not None:
            source_metadata["crop"] = str(crop_path)
        regions.append(
            RecoveredRegion(
                order=int(raw.get("order", len(regions))),
                kind=kind,
                bounds=bounds,
                prepared_bounds=prepared_bounds,
                qwen_text=raw.get("text"),
                text=text,
                source=source_metadata,
                recognizer={
                    "inference_ms": round(inference_ms, 3),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
            )
        )
    response_metadata = getattr(interpreter, "response_metadata", {})
    total_ms = (time.perf_counter() - recovery_started) * 1000
    runtime = {
        "vlm_model": getattr(interpreter, "model", None),
        "vlm_ms": round(qwen_ms, 3),
        "vlm_dimensions": {
            "width": prepared.prepared_dimensions[0],
            "height": prepared.prepared_dimensions[1],
        },
        "recognizer": recognizer_metadata,
        "benchmark": {
            "model": getattr(interpreter, "model", None),
            "runtime": "ollama",
            "preparation_dimensions": {
                "width": prepared.prepared_dimensions[0],
                "height": prepared.prepared_dimensions[1],
            },
            "source_dimensions": {"width": source.width, "height": source.height},
            "scale": prepared.scale,
            "preparation_ms": round(prepared.preparation_ms, 3),
            "qwen_ms": round(qwen_ms, 3),
            "prompt_tokens": response_metadata.get("prompt_tokens"),
            "output_tokens": response_metadata.get("output_tokens"),
            "region_count": len(regions),
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
    )


def format_recovery(page: RecoveredPage) -> str:
    return json.dumps(asdict(page), indent=2)


def format_text_recovery(page: RecoveredPage) -> str:
    """Project only recovered text, preserving model spelling and order."""
    lines: list[str] = []
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
    markdown_lines: list[str] = []
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
            label = region.qwen_text or f"Figure {figure_count:02d}"
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

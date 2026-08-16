"""Region-first repeated handwriting observations (experimental only)."""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image, UnidentifiedImageError

from sibyl.transform import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_QWEN_MODEL,
    PreparedVlmImage,
    map_prepared_bounds,
    pad_normalized_bounds,
    prepare_page_image_with_metadata,
    qwen_bbox_to_normalized,
)

DEFAULT_RUNS = 5
DEFAULT_OUTPUT = Path(".sibyl/experiments/transcription-reread.json")
LOCALIZATION_NUM_PREDICT = 512
REGIONAL_NUM_PREDICT = 256
LOCALIZATION_PROMPT = (
    "Identify handwritten text regions in this image. Return only the requested JSON structure: "
    "text_regions containing bbox_2d with exactly four numeric values [x1, y1, x2, y2] in "
    "Qwen's 0..1000 coordinate space. Use no prose, OCR, reasoning, commentary, or extra values."
)
LOCALIZATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                "required": ["bbox_2d"],
            },
        }
    },
    "required": ["text_regions"],
}
REGIONAL_PROMPT = (
    "Read the handwritten text in this image exactly as written. Return only the text. "
    "Do not interpret diagrams or surrounding page content. Preserve uncertainty rather "
    "than inventing text."
)
REGIONAL_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}
PADDING_PROPORTION = 0.05


class TextRegionLocalizer(Protocol):
    model: str

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


class RegionalReader(Protocol):
    model: str

    def read(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


def _image_data(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _query_ollama(
    *,
    model: str,
    base_url: str,
    prompt: str,
    schema: dict[str, Any],
    image: Image.Image,
    num_predict: int,
    observer: Callable[[dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"num_predict": num_predict},
        "keep_alive": 0,
        "messages": [{"role": "user", "content": prompt, "images": [_image_data(image)]}],
    }
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to query Ollama/Qwen ({model}): {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("Ollama/Qwen returned a non-object response")
    if observer is not None:
        observer(body)
    return body, (time.perf_counter() - started) * 1000


def _message_json(body: dict[str, Any]) -> Any:
    message = body.get("message", {})
    if not isinstance(message, dict):
        return None
    for field in ("content", "thinking"):
        candidate = message.get(field)
        if isinstance(candidate, str):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


class OllamaTextRegionLocalizer:
    """Dedicated experimental text localizer; never invokes drawing localization."""

    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        self.base_url = (base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip(
            "/"
        )
        self._observer = observer

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        body, duration_ms = _query_ollama(
            model=self.model,
            base_url=self.base_url,
            prompt=LOCALIZATION_PROMPT,
            schema=LOCALIZATION_SCHEMA,
            image=image,
            num_predict=LOCALIZATION_NUM_PREDICT,
            observer=self._observer,
        )
        parsed = _message_json(body)
        truncated = body.get("done_reason") == "length"
        if not isinstance(parsed, dict) or not isinstance(parsed.get("text_regions"), list):
            return {
                "status": "truncated_response" if truncated else "invalid_response",
                "error": (
                    "localization response was truncated" if truncated else "missing text_regions"
                ),
                "raw_response": body,
            }, duration_ms
        if truncated:
            return {
                "status": "truncated_response",
                "error": "localization response was truncated",
                "text_regions": parsed["text_regions"],
                "raw_response": body,
            }, duration_ms
        return {"text_regions": parsed["text_regions"]}, duration_ms

    def release(self) -> None:
        return None


class OllamaRegionalReader:
    """Independent minimal OCR request for one source-resolution crop."""

    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        self.base_url = (base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip(
            "/"
        )
        self._observer = observer

    def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        body, duration_ms = _query_ollama(
            model=self.model,
            base_url=self.base_url,
            prompt=REGIONAL_PROMPT,
            schema=REGIONAL_SCHEMA,
            image=image,
            num_predict=REGIONAL_NUM_PREDICT,
            observer=self._observer,
        )
        parsed = _message_json(body)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("text"), str):
            return {
                "status": "invalid_response",
                "error": "missing text",
                "raw_response": body,
            }, duration_ms
        return {"text": parsed["text"]}, duration_ms

    def release(self) -> None:
        return None


def requested_runs(value: int | None = None) -> int:
    configured = value
    if configured is None:
        raw = os.environ.get("SIBYL_TRANSCRIPTION_REREAD_RUNS", str(DEFAULT_RUNS))
        try:
            configured = int(raw)
        except ValueError as error:
            raise ValueError(
                "SIBYL_TRANSCRIPTION_REREAD_RUNS must be a positive integer"
            ) from error
    if configured <= 0:
        raise ValueError("regional reread runs must be a positive integer")
    return configured


def _validate_bbox(value: Any) -> tuple[bool, str | None]:
    if not isinstance(value, list) or len(value) != 4:
        return False, "bbox must be an array of four coordinates"
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    ):
        return False, "bbox coordinates must be numeric"
    if not all(0 <= float(item) <= 1000 for item in value):
        return False, "bbox coordinates must be within qwen_0_1000"
    left, top, right, bottom = (float(item) for item in value)
    if right <= left or bottom <= top:
        return False, "bbox must have positive area and non-inverted bounds"
    return True, None


def validate_regions(regions: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not isinstance(regions, list):
        return [], [{"index": None, "bbox": regions, "reason": "text_regions must be an array"}]
    for index, region in enumerate(regions):
        bbox = region.get("bbox_2d") if isinstance(region, dict) else None
        valid, reason = _validate_bbox(bbox)
        if valid:
            accepted.append(
                {"index": index, "bbox_2d": [float(item) for item in cast(list[Any], bbox)]}
            )
        else:
            rejected.append({"index": index, "bbox": bbox, "reason": reason})
    return accepted, rejected


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def deduplicate_regions(
    regions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for region in regions:
        if any(_iou(region["bbox_2d"], prior["bbox_2d"]) >= 0.5 for prior in kept):
            rejected.append(
                {
                    "index": region["index"],
                    "bbox": region["bbox_2d"],
                    "reason": "duplicate_or_overlapping_bbox",
                }
            )
        else:
            kept.append(region)
    return kept, rejected


def _source_crop(
    source: Image.Image,
    prepared: PreparedVlmImage,
    bbox: list[float],
    output_path: Path,
    region_id: str,
) -> dict[str, Any]:
    normalized = qwen_bbox_to_normalized(cast(tuple[float, float, float, float], tuple(bbox)))
    padded = pad_normalized_bounds(normalized, proportion=PADDING_PROPORTION)
    prepared_bounds = tuple(
        value * dimension
        for value, dimension in zip(
            padded,
            (prepared.prepared_dimensions[0], prepared.prepared_dimensions[1]) * 2,
            strict=True,
        )
    )
    bounds = map_prepared_bounds(
        cast(tuple[float, float, float, float], prepared_bounds),
        prepared.prepared_dimensions,
        source.size,
    )
    crop = source.crop((bounds.left, bounds.top, bounds.right, bounds.bottom)).convert("RGB")
    crop_path = output_path.parent / "transcription-reread" / f"{region_id}.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path, format="PNG")
    return {
        "path": str(crop_path),
        "width": crop.width,
        "height": crop.height,
        "source_bbox": asdict(bounds),
        "source_coordinate_space": "source",
        "padding": {"proportion": PADDING_PROPORTION, "normalized_bbox": list(padded)},
        "mapping": {
            "from": "qwen_0_1000",
            "to": "source",
            "prepared_dimensions": {
                "width": prepared.prepared_dimensions[0],
                "height": prepared.prepared_dimensions[1],
            },
            "source_dimensions": {"width": source.width, "height": source.height},
        },
        "image": crop,
    }


def _controls(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "think": False,
        "num_predict": 256,
        "stream": False,
        "keep_alive": 0,
        "temperature": "unspecified (Ollama/model default)",
        "top_p": "unspecified (Ollama/model default)",
        "seed": "unspecified (Ollama/model default)",
        "prompt": REGIONAL_PROMPT,
        "schema": REGIONAL_SCHEMA,
    }


def run_reread_experiment(
    image_path: Path,
    *,
    runs: int | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    localizer_factory: Callable[[Callable[[dict[str, Any]], None]], TextRegionLocalizer]
    | None = None,
    reader_factory: Callable[[Callable[[dict[str, Any]], None]], RegionalReader] | None = None,
) -> dict[str, Any]:
    requested = requested_runs(runs)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as source_file:
            source = source_file.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error
    prepared = prepare_page_image_with_metadata(source)
    raw_localization: Any = None

    def observe_localization(response: dict[str, Any]) -> None:
        nonlocal raw_localization
        raw_localization = response

    localizer = (
        localizer_factory(observe_localization)
        if localizer_factory
        else OllamaTextRegionLocalizer(observer=observe_localization)
    )
    try:
        try:
            localization, localization_ms = localizer.localize(prepared.image)
        except (RuntimeError, ValueError) as error:
            localization, localization_ms = {"status": "failed", "error": str(error)}, 0.0
    finally:
        localizer.release()

    accepted, rejected = validate_regions(localization.get("text_regions"))
    accepted, duplicate_rejections = deduplicate_regions(accepted)
    rejected.extend(duplicate_rejections)
    accepted.sort(key=lambda item: item["index"])
    localization_status = localization.get("status")
    if localization_status is None:
        localization_status = (
            "ok" if isinstance(localization.get("text_regions"), list) else "invalid_response"
        )
    artifact: dict[str, Any] = {
        "experiment": "transcription_reread",
        "source": str(image_path),
        "page": {
            "focus": prepared.focus,
            "source_dimensions": {"width": source.width, "height": source.height},
            "prepared_dimensions": {
                "width": prepared.prepared_dimensions[0],
                "height": prepared.prepared_dimensions[1],
            },
            "prepared_image_hash": _prepared_hash(prepared),
        },
        "localization": {
            "status": localization_status,
            "error": localization.get("error"),
            "raw_response": raw_localization or localization.get("raw_response"),
            "duration_ms": round(localization_ms, 3),
            "model_coordinate_space": "qwen_0_1000",
            "rejected_regions": rejected,
            "request_controls": {
                "model": getattr(localizer, "model", DEFAULT_QWEN_MODEL),
                "think": False,
                "stream": False,
                "keep_alive": 0,
                "num_predict": LOCALIZATION_NUM_PREDICT,
                "temperature": "unspecified (Ollama/model default)",
                "top_p": "unspecified (Ollama/model default)",
                "seed": "unspecified (Ollama/model default)",
                "prompt": LOCALIZATION_PROMPT,
                "schema": LOCALIZATION_SCHEMA,
            },
        },
        "regions": [],
        "runs_requested": requested,
        "request_controls": {},
    }
    for number, located in enumerate(accepted, start=1):
        region_id = f"region-{number:02d}"
        crop_info = _source_crop(source, prepared, located["bbox_2d"], output_path, region_id)
        crop_image = cast(Image.Image, crop_info.pop("image"))
        reads: list[dict[str, Any]] = []
        reader_raw: Any = None

        def observe_reader(response: dict[str, Any]) -> None:
            nonlocal reader_raw
            reader_raw = response

        reader = (
            reader_factory(observe_reader)
            if reader_factory
            else OllamaRegionalReader(observer=observe_reader)
        )
        try:
            for run in range(1, requested + 1):
                reader_raw = None
                started = time.perf_counter()
                try:
                    result, duration_ms = reader.read(crop_image)
                    raw = reader_raw if reader_raw is not None else result.get("raw_response")
                    if result.get("status") == "invalid_response" or not isinstance(
                        result.get("text"), str
                    ):
                        reads.append(
                            {
                                "run": run,
                                "status": "invalid_response",
                                "text": None,
                                "raw_response": raw,
                                "error": result.get("error", "missing text"),
                                "duration_ms": round(duration_ms, 3),
                            }
                        )
                    else:
                        reads.append(
                            {
                                "run": run,
                                "status": "ok",
                                "text": result["text"],
                                "raw_response": raw,
                                "duration_ms": round(duration_ms, 3),
                            }
                        )
                except (RuntimeError, ValueError) as error:
                    reads.append(
                        {
                            "run": run,
                            "status": "failed",
                            "text": None,
                            "raw_response": reader_raw,
                            "error": str(error),
                            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                        }
                    )
        finally:
            reader.release()
            crop_image.close()
        distinct = list(dict.fromkeys(item["text"] for item in reads if item["status"] == "ok"))
        artifact["regions"].append(
            {
                "region_id": region_id,
                "model_bbox": located["bbox_2d"],
                "model_coordinate_space": "qwen_0_1000",
                **crop_info,
                "reads": reads,
                "distinct_readings": distinct,
                "stable": len(distinct) == 1 and len(distinct) > 0,
            }
        )
        artifact["request_controls"] = _controls(getattr(reader, "model", DEFAULT_QWEN_MODEL))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def _prepared_hash(prepared: PreparedVlmImage) -> str:
    import hashlib

    output = BytesIO()
    prepared.image.save(output, format="PNG")
    return hashlib.sha256(output.getvalue()).hexdigest()


def format_reread_result(result: dict[str, Any]) -> str:
    page = result["page"]
    lines = [
        f"experiment: {result['experiment']}",
        f"source: {result['source']}",
        "page image: "
        f"{page['prepared_dimensions']['width']}x{page['prepared_dimensions']['height']}",
        f"page image hash: {page['prepared_image_hash']}",
        "",
        "localized text regions: "
        f"{len(result['regions']) + len(result['localization']['rejected_regions'])}",
        f"accepted regions: {len(result['regions'])}",
        f"rejected regions: {len(result['localization']['rejected_regions'])}",
    ]
    for region in result["regions"]:
        lines.extend(
            [
                f"\n{region['region_id']}:",
                f"  source bbox: {region['source_bbox']}",
                f"  crop: {region['path']}",
            ]
        )
        lines.extend(
            f"  Run {read['run']}: {read.get('text') or read.get('error')} [{read['status']}]"
            for read in region["reads"]
        )
        lines.extend(
            [
                f"  distinct readings: {len(region['distinct_readings'])}",
                f"  stable: {str(region['stable']).lower()}",
            ]
        )
    return "\n".join(lines)

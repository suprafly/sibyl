"""Collect independent rereads of page regions that disagree across observations."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from sibyl.experiments.transcription_variance import (
    PageTranscriber,
    VarianceResult,
    prepared_image_hash,
    run_variance_experiment,
)
from sibyl.transform import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_QWEN_MODEL,
    PreparedVlmImage,
    map_prepared_bounds,
    pad_normalized_bounds,
    qwen_bbox_to_normalized,
)

DEFAULT_REREADS = 3
DEFAULT_OUTPUT = Path(".sibyl/experiments/transcription-reread.json")
LOCALIZATION_PROMPT = (
    "Locate the handwritten text lines on this page in reading order. Return only JSON. "
    "For each text line, return its bounding box as bbox_2d in Qwen's 0 to 1000 "
    "coordinate space. Do not include drawings, arrows, diagram strokes, or other graphics."
)
LOCALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "text_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["bbox_2d"],
            },
        }
    },
    "required": ["text_regions"],
}
TARGETED_PROMPT = (
    "Read the handwritten text in this image exactly as written. Preserve uncertainty "
    "rather than inventing a word. Do not interpret diagrams or surrounding page content. "
    "Return only JSON matching the schema."
)
TARGETED_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


class TextRegionLocalizer(Protocol):
    model: str

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


class TargetedRereader(Protocol):
    model: str

    def reread(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

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
    observer: Callable[[dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"num_predict": 256},
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
    """Experimental ordered text-box request; it is not the drawing localizer."""

    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")
        self._observer = observer

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        body, duration_ms = _query_ollama(
            model=self.model,
            base_url=self.base_url,
            prompt=LOCALIZATION_PROMPT,
            schema=LOCALIZATION_SCHEMA,
            image=image,
            observer=self._observer,
        )
        parsed = _message_json(body)
        if not isinstance(parsed, dict) or not _valid_regions(parsed.get("text_regions")):
            return {
                "status": "failure",
                "error": "targeted localization returned no valid text-region boxes",
                "raw_response": body,
            }, duration_ms
        return {"text_regions": parsed["text_regions"]}, duration_ms

    def release(self) -> None:
        return None


class OllamaTargetedRereader:
    """Experimental single-crop reread request with no candidate context."""

    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        configured_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = configured_url.rstrip("/")
        self._observer = observer

    def reread(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        body, duration_ms = _query_ollama(
            model=self.model,
            base_url=self.base_url,
            prompt=TARGETED_PROMPT,
            schema=TARGETED_SCHEMA,
            image=image,
            observer=self._observer,
        )
        parsed = _message_json(body)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("text"), str):
            return {
                "status": "failure",
                "error": "targeted reread returned no valid text",
                "raw_response": body,
            }, duration_ms
        return {"text": parsed["text"]}, duration_ms

    def release(self) -> None:
        return None


def _valid_regions(regions: Any) -> bool:
    if not isinstance(regions, list):
        return False
    for region in regions:
        if not isinstance(region, dict):
            return False
        bbox = region.get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
            or not all(0 <= float(value) <= 1000 for value in bbox)
        ):
            return False
    return True


def _disagreements(page_result: VarianceResult) -> list[dict[str, Any]]:
    successful = [run for run in page_result.runs if run.status == "ok" and run.lines is not None]
    if not successful:
        return []
    line_count = max(len(run.lines or []) for run in successful)
    disagreements: list[dict[str, Any]] = []
    for index in range(line_count):
        values = [
            (run.lines or [])[index]
            for run in successful
            if index < len(run.lines or [])
        ]
        candidates = list(dict.fromkeys(values))
        if len(candidates) > 1:
            disagreements.append(
                {
                    "region_id": f"region-{len(disagreements) + 1:02d}",
                    "line_index": index,
                    "page_candidates": candidates,
                }
            )
    return disagreements


def _source_crop(
    source: Image.Image,
    prepared: PreparedVlmImage,
    bbox: list[float],
    output_path: Path,
    region_id: str,
) -> dict[str, Any]:
    normalized = qwen_bbox_to_normalized(cast(tuple[float, float, float, float], tuple(bbox)))
    padded = pad_normalized_bounds(normalized, proportion=0.05)
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
        "padding": 0.05,
        "width": crop.width,
        "height": crop.height,
        "source_bounds": asdict(bounds),
        "coordinate_space": "source",
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
    }


def run_reread_experiment(
    image_path: Path,
    *,
    runs: int | None = None,
    rereads: int = DEFAULT_REREADS,
    output_path: Path = DEFAULT_OUTPUT,
    localizer_factory: Callable[[Callable[[dict[str, Any]], None]], TextRegionLocalizer]
    | None = None,
    rereader_factory: Callable[[Callable[[dict[str, Any]], None]], TargetedRereader]
    | None = None,
    interpreter_factory: Callable[[Callable[[dict[str, Any]], None]], PageTranscriber]
    | None = None,
) -> dict[str, Any]:
    if rereads <= 0:
        raise ValueError("targeted rereads must be a positive integer")
    prepared_capture: dict[str, Any] = {}

    def capture(source: Image.Image, prepared: PreparedVlmImage) -> None:
        prepared_capture["source"] = source
        prepared_capture["prepared"] = prepared

    page_result = run_variance_experiment(
        image_path,
        runs=runs,
        output_path=output_path,
        prepared_observer=capture,
        write_output=False,
        interpreter_factory=interpreter_factory,
    )
    source = cast(Image.Image, prepared_capture["source"])
    prepared = cast(PreparedVlmImage, prepared_capture["prepared"])
    disagreements = _disagreements(page_result)
    artifact: dict[str, Any] = {
        "experiment": "transcription_reread",
        "source": page_result.source,
        "source_dimensions": {"width": source.width, "height": source.height},
        "page_focus": page_result.page_focus,
        "prepared_dimensions": page_result.prepared_dimensions,
        "prepared_image_hash": prepared_image_hash(prepared),
        "page_runs_requested": page_result.runs_requested,
        "page_observations": asdict(page_result),
        "disagreements": disagreements,
        "localization": {"status": "not_run"},
        "targeted_rereads": rereads,
    }
    if not disagreements:
        artifact["localization"] = {"status": "not_needed"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        return artifact

    raw_localization: Any = None

    def observe_localization(response: dict[str, Any]) -> None:
        nonlocal raw_localization
        raw_localization = response

    localizer = (
        localizer_factory(observe_localization)
        if localizer_factory is not None
        else OllamaTextRegionLocalizer(observer=observe_localization)
    )
    try:
        try:
            localization, localization_ms = localizer.localize(prepared.image)
        except (RuntimeError, ValueError) as error:
            localization = {"status": "failure", "error": str(error)}
            localization_ms = 0.0
    finally:
        localizer.release()
    regions = localization.get("text_regions", [])
    if localization.get("status") == "failure" or not _valid_regions(regions):
        artifact["localization"] = {
            "status": "unavailable",
            "message": "candidate disagreement detected; targeted localization unavailable",
            "error": localization.get("error"),
            "raw_response": raw_localization or localization.get("raw_response"),
            "duration_ms": round(localization_ms, 3),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        return artifact

    ordered_regions = sorted(
        regions, key=lambda region: int(region.get("order", regions.index(region)))
    )
    if any(disagreement["line_index"] >= len(ordered_regions) for disagreement in disagreements):
        artifact["localization"] = {
            "status": "unavailable",
            "message": "candidate disagreement detected; targeted localization unavailable",
            "error": "localization did not cover every disagreeing text line",
            "raw_response": raw_localization,
            "duration_ms": round(localization_ms, 3),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        return artifact

    artifact["localization"] = {
        "status": "success",
        "coordinate_space": "qwen_0_1000",
        "raw_response": raw_localization,
        "duration_ms": round(localization_ms, 3),
    }
    for disagreement in disagreements:
        region = ordered_regions[disagreement["line_index"]]
        bbox = cast(list[float], region["bbox_2d"])
        disagreement["bbox"] = {"space": "qwen_0_1000", "bbox_2d": bbox}
        disagreement["crop"] = _source_crop(
            source, prepared, bbox, output_path, disagreement["region_id"]
        )
        disagreement["rereads"] = []
        raw_reread: Any = None

        def observe_reread(response: dict[str, Any]) -> None:
            nonlocal raw_reread
            raw_reread = response

        rereader = (
            rereader_factory(observe_reread)
            if rereader_factory is not None
            else OllamaTargetedRereader(observer=observe_reread)
        )
        crop_image = Image.open(disagreement["crop"]["path"]).convert("RGB")
        try:
            for number in range(1, rereads + 1):
                raw_reread = None
                started = time.perf_counter()
                try:
                    reread, duration_ms = rereader.reread(crop_image)
                    raw = raw_reread or reread.get("raw_response")
                    if reread.get("status") == "failure":
                        disagreement["rereads"].append(
                            {
                                "run": number,
                                "status": "invalid_response",
                                "text": None,
                                "raw_response": raw,
                                "error": reread.get("error"),
                                "duration_ms": round(duration_ms, 3),
                            }
                        )
                    else:
                        disagreement["rereads"].append(
                            {
                                "run": number,
                                "status": "ok",
                                "text": reread.get("text"),
                                "raw_response": raw,
                                "duration_ms": round(duration_ms, 3),
                            }
                        )
                except (RuntimeError, ValueError) as error:
                    disagreement["rereads"].append(
                        {
                            "run": number,
                            "status": "failed",
                            "text": None,
                            "raw_response": raw_reread,
                            "error": str(error),
                            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                        }
                    )
        finally:
            rereader.release()
            crop_image.close()
    artifact["targeted_request_controls"] = _controls(
        getattr(rereader, "model", page_result.model)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def format_reread_result(result: dict[str, Any]) -> str:
    lines = [
        f"experiment: {result['experiment']}",
        f"source: {result['source']}",
        f"page runs: {result['page_runs_requested']}",
        "page image: "
        f"{result['prepared_dimensions']['width']}x{result['prepared_dimensions']['height']}",
        f"page image hash: {result['prepared_image_hash']}",
        "",
        f"Disagreement regions: {len(result['disagreements'])}",
    ]
    for disagreement in result["disagreements"]:
        lines.extend([f"\n{disagreement['region_id']}:", "  page candidates:"])
        lines.extend(f"    {candidate}" for candidate in disagreement["page_candidates"])
        if "crop" not in disagreement:
            lines.append("  targeted localization unavailable")
            continue
        lines.append(f"  crop: {disagreement['crop']['path']}")
        lines.append("  rereads:")
        for reread in disagreement["rereads"]:
            lines.append(
                f"    Run {reread['run']} [{reread['status']}]: "
                f"{reread.get('text') or reread.get('error')}"
            )
    if result["localization"]["status"] == "unavailable":
        lines.append("\ncandidate disagreement detected")
        lines.append("targeted localization unavailable")
    return "\n".join(lines)

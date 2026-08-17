"""Measure whether verified BOOX stroke exemplars help Qwen recognition."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

from sibyl.experiments.boox_strokes import inspect_boox_strokes
from sibyl.experiments.handwriting_exemplars import (
    ExemplarReader,
    OllamaExemplarReader,
    _read_configuration,
)
from sibyl.experiments.handwriting_preprocess import _targets, evaluate_candidate
from sibyl.experiments.qwen_recognition_knobs import (
    ISOLATED_PROMPT,
    extract_recognition_text,
)
from sibyl.transform import DEFAULT_QWEN_MODEL

DEFAULT_NOTE = Path("samples/Grafting 101.note")
DEFAULT_IMAGE = Path("samples/Grafting-101-page-004.png")
DEFAULT_OUTPUT = Path(".sibyl/experiments/boox-recognition.json")
DEFAULT_REREAD = Path(".sibyl/experiments/transcription-reread.json")
DEFAULT_COMPARE = Path(".sibyl/experiments/trocr-compare.json")
DEFAULT_RUNS = 5
BOOX_RECOGNITION_NUM_PREDICT = 1024
PAGE = 4
NATIVE_SIZE = (1404, 1872)
CONDITIONS = (
    "baseline",
    "native-render",
    "native-exemplar",
    "multi-exemplar",
    "leave-one-region-out",
)
CRITICAL_LINES = (
    "region-02-line-01",
    "region-02-line-02",
    "region-02-line-03",
    "region-02-line-04",
    "region-03-line-01",
)
NATIVE_PROMPT = (
    "REFERENCE IMAGES appear first, in the listed reference order. TARGET IMAGE appears last.\n"
    "Use the reference images only to understand this writer's handwriting style and glyph forms.\n"
    "Do not copy words from references unless the target visibly contains them.\n"
    "Transcribe only the TARGET IMAGE. Return only its transcription.\n"
    "Do not describe the images or infer text that is not visible."
)

ReaderFactory = Callable[[Callable[[dict[str, Any]], None]], ExemplarReader]


def selected_conditions(value: str | None) -> tuple[str, ...]:
    """Return requested conditions in canonical order for reproducible runs."""
    if value is None:
        return CONDITIONS
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("--conditions must contain at least one condition")
    if len(set(requested)) != len(requested):
        raise ValueError("--conditions must not contain duplicates")
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown BOOX recognition conditions: {', '.join(unknown)}")
    return tuple(condition for condition in CONDITIONS if condition in requested)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bbox_tuple(value: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        values = tuple(
            round(float(value[key])) for key in ("left", "top", "right", "bottom")
        )
        return (values[0], values[1], values[2], values[3])
    except (KeyError, TypeError, ValueError):
        return None


def _intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def _map_bbox(
    source_bbox: tuple[int, int, int, int], source_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    sx = NATIVE_SIZE[0] / source_size[0]
    sy = NATIVE_SIZE[1] / source_size[1]
    values = tuple(
        round(value * (sx if index % 2 == 0 else sy))
        for index, value in enumerate(source_bbox)
    )
    return (values[0], values[1], values[2], values[3])


def _stroke_points(stroke: dict[str, Any]) -> list[dict[str, Any]]:
    points = stroke.get("native_points") or stroke.get("points") or []
    return [point for point in points if isinstance(point, dict)]


def _stroke_bounds(stroke: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = stroke.get("native_bounds") or stroke.get("bounds")
    return _bbox_tuple(value)


def select_native_strokes(
    strokes: list[dict[str, Any]],
    native_bbox: tuple[int, int, int, int] | None = None,
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select complete native strokes in decoder order, never individual points."""
    selected: list[dict[str, Any]] = []
    for stroke in sorted(strokes, key=lambda item: item.get("order", 0)):
        stroke_id = stroke.get("stroke_id")
        if exclude_ids and stroke_id in exclude_ids:
            continue
        bounds = _stroke_bounds(stroke)
        if native_bbox is not None and (bounds is None or not _intersects(bounds, native_bbox)):
            continue
        selected.append(stroke)
    return selected


def render_native_reference(
    path: Path,
    strokes: list[dict[str, Any]],
    *,
    native_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Render native coordinates with fixed geometry and record exact provenance."""
    if native_bbox is None:
        origin = (0, 0)
        size = NATIVE_SIZE
    else:
        left, top, right, bottom = native_bbox
        origin = (left, top)
        size = (max(1, right - left), max(1, bottom - top))
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = _stroke_points(stroke)
        pixels = [
            (round(point["x"] - origin[0]), round(point["y"] - origin[1]))
            for point in points
        ]
        if len(pixels) > 1:
            draw.line(pixels, fill=(0, 0, 0), width=2, joint="curve")
        elif pixels:
            x, y = pixels[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "dimensions": {"width": size[0], "height": size[1]},
        "native_origin": {"x": origin[0], "y": origin[1]},
        "stroke_ids": [stroke.get("stroke_id") for stroke in strokes],
        "point_counts": [
            int(stroke.get("point_count", len(_stroke_points(stroke)))) for stroke in strokes
        ],
        "rendering": {
            "background": "white",
            "stroke_width": 2,
            "coordinate_transform": "identity",
            "crop": "native_bbox" if native_bbox is not None else "full_page",
        },
    }


def _line_catalog(path: Path, image_path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Line artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("source") != str(image_path):
        raise ValueError("transcription-reread artifact source does not match page image")
    lines: list[dict[str, Any]] = []
    for region in artifact.get("regions", []):
        for line in region.get("line_localization", {}).get("regions", []):
            line_id = line.get("line_id")
            bbox = _bbox_tuple(line.get("source_bbox"))
            if isinstance(line_id, str) and bbox is not None:
                lines.append(
                    {
                        "reference_id": line_id,
                        "source_bbox": line.get("source_bbox"),
                        "bbox": bbox,
                        "source_artifact": str(path),
                    }
                )
    return sorted(lines, key=lambda item: item["reference_id"])


def _load_review(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Review file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        entries: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for line in raw.splitlines():
            item = line.strip()
            if item.startswith("- target_id:"):
                current = {"target_id": item.split(":", 1)[1].strip().strip("\"'")}
                entries.append(current)
            elif current is not None and ":" in item:
                key, text = item.split(":", 1)
                current[key.strip()] = text.strip().strip("\"'")
        value = {"targets": entries}
    if not isinstance(value, dict) or not isinstance(value.get("targets"), list):
        raise ValueError("review requires a targets list")
    result: dict[str, str] = {}
    for item in value["targets"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("target_id"), str)
            or not isinstance(item.get("transcription"), str)
            or item.get("confirmed") is not True
        ):
            raise ValueError("each review target requires confirmed transcription metadata")
        if item["target_id"] in result:
            raise ValueError(f"duplicate review target: {item['target_id']}")
        result[item["target_id"]] = item["transcription"]
    return result


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+|[^\w\s]", value, flags=re.UNICODE)


def evaluate_reading(reading: str, truth: str) -> dict[str, Any]:
    """Evaluate observed text without replacing the original reading."""
    candidate_tokens = _tokens(reading)
    truth_tokens = _tokens(truth)
    remaining = list(candidate_tokens)
    overlap = 0
    for token in truth_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    metrics = evaluate_candidate(reading, truth)
    return {
        **metrics,
        "raw_exact_match": reading == truth,
        "word_overlap": overlap / len(truth_tokens) if truth_tokens else 0.0,
        "unresolved_tokens": [token for token in truth_tokens if token not in candidate_tokens],
        "reading": reading,
        "ground_truth": truth,
    }


def _condition_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    readings = [reading for reading in analysis["readings"] if isinstance(reading, str)]
    distribution = Counter(readings)
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    stable = ordered[0][0] if ordered and ordered[0][1] > len(readings) / 2 else None
    return {
        **analysis,
        "stable_reading": stable,
        "candidate_distribution": [
            {"reading": reading, "count": count, "frequency": count / len(readings)}
            for reading, count in ordered
        ],
    }


def _truncated_thinking_evidence(raw_response: Any) -> str | None:
    """Extract truncated thinking for analysis without making it a reading."""
    if not isinstance(raw_response, dict):
        return None
    message = raw_response.get("message")
    if not isinstance(message, dict):
        return None
    thinking = message.get("thinking")
    if not thinking:
        return None
    return extract_recognition_text({"message": {"content": "", "thinking": thinking}})


def _add_truncated_evidence(analysis: dict[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    for item in analysis["runs"]:
        value = (
            _truncated_thinking_evidence(item.get("raw_response"))
            if item.get("status") == "truncated_response"
            else None
        )
        if value is not None:
            item["truncated_evidence"] = value
            evidence.append(value)
    analysis["truncated_evidence"] = evidence
    return analysis


def _verified_page(note_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sibyl-boox-recognition-") as directory:
        result = inspect_boox_strokes(note_path, page=PAGE, output=Path(directory) / "page")
    if result["selected_page"]["note_page"] != PAGE or tuple(
        result["reconstruction"]["native_dimensions"]
    ) != NATIVE_SIZE:
        raise ValueError("BOOX recognition requires the verified page-4 native dimensions")
    points = sum(
        int(stroke.get("point_count", len(_stroke_points(stroke))))
        for stroke in result["strokes"]
    )
    if len(result["strokes"]) != 167 or points != 17272:
        raise ValueError(
            "BOOX recognition requires the verified 167-stroke/17,272-point page-4 decode"
        )
    if any(stroke.get("shape_association") is False for stroke in result["strokes"]):
        raise ValueError("BOOX recognition requires verified stroke/point associations")
    return result


def run_boox_recognition(
    image_path: Path = DEFAULT_IMAGE,
    *,
    note_path: Path = DEFAULT_NOTE,
    regions: str | None = None,
    lines: str | None = None,
    runs: int = DEFAULT_RUNS,
    num_predict: int = BOOX_RECOGNITION_NUM_PREDICT,
    conditions: str | None = None,
    review_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    reread_path: Path = DEFAULT_REREAD,
    compare_path: Path = DEFAULT_COMPARE,
    reader_factory: ReaderFactory | None = None,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    if num_predict <= 0:
        raise ValueError("num_predict must be positive")
    requested_conditions = selected_conditions(conditions)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not note_path.is_file():
        raise FileNotFoundError(f"Source file not found: {note_path}")
    if lines is None and regions is None:
        targets = _targets(
            image_path,
            regions=None,
            lines=",".join(CRITICAL_LINES),
            crop_path=None,
            compare_path=compare_path,
            reread_path=reread_path,
        )
        targets.extend(
            _targets(
                image_path,
                regions="region-05",
                lines=None,
                crop_path=None,
                compare_path=compare_path,
                reread_path=reread_path,
            )
        )
        targets = list({target["target_id"]: target for target in targets}.values())
    else:
        targets = _targets(
            image_path,
            regions=regions,
            lines=lines,
            crop_path=None,
            compare_path=compare_path,
            reread_path=reread_path,
        )
    native = _verified_page(note_path)
    source_size = (Image.open(image_path).width, Image.open(image_path).height)
    catalog = _line_catalog(reread_path, image_path)
    review = _load_review(review_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = output_path.parent / "boox-recognition"

    def checkpoint() -> None:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_path)

    controls = {
        "model": DEFAULT_QWEN_MODEL,
        "temperature": "unspecified (Ollama/model default)",
        "top_p": "unspecified (Ollama/model default)",
        "seed": "unspecified (Ollama/model default)",
        "num_predict": num_predict,
        "think": False,
        "stream": False,
        "keep_alive": 0,
    }
    artifact: dict[str, Any] = {
        "experiment": "boox_recognition",
        "source": str(image_path),
        "source_sha256": _sha256(image_path),
        "source_dimensions": {"width": source_size[0], "height": source_size[1]},
        "native_source": {
            "note": str(note_path),
            "note_sha256": _sha256(note_path),
            "page": PAGE,
            "dimensions": {"width": NATIVE_SIZE[0], "height": NATIVE_SIZE[1]},
            "stroke_count": len(native["strokes"]),
            "point_count": sum(
                int(stroke.get("point_count", len(_stroke_points(stroke))))
                for stroke in native["strokes"]
            ),
            "coordinate_transform": "identity",
            "associations": "verified",
        },
        "runs_requested": runs,
        "conditions_requested": list(requested_conditions),
        "status": "running",
        "completed_results": [],
        "request_controls": controls,
        "targets": [],
        "evaluation_review": str(review_path) if review_path else None,
        "results": [],
        "output": str(output_path),
    }
    checkpoint()
    strokes = native["strokes"]
    for target in targets:
        target_id = target["target_id"]
        target_bbox = _bbox_tuple(target.get("source_bbox"))
        target_native_bbox = _map_bbox(target_bbox, source_size) if target_bbox else None
        target_record = {
            "target_id": target_id,
            "kind": target["kind"],
            "path": str(target["path"]),
            "sha256": _sha256(Path(target["path"])),
            "source_bbox": target.get("source_bbox"),
            "native_bbox": target_native_bbox,
        }
        artifact["targets"].append(target_record)
        checkpoint()
        other_lines = [
            line
            for line in catalog
            if target_bbox is None
            or not _intersects(target_bbox, line["bbox"])
            or (line["reference_id"] == target_id and target["kind"] != "line")
        ]
        other_lines = [line for line in other_lines if line["reference_id"] != target_id]
        specs: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "native-render": [{"reference_id": "page-004-native", "kind": "full-page"}],
            "native-exemplar": other_lines[:1],
            "multi-exemplar": other_lines,
            "leave-one-region-out": other_lines,
        }
        for condition in requested_conditions:
            references = specs[condition]
            reference_records: list[dict[str, Any]] = []
            if condition == "baseline":
                pass
            elif condition == "native-render":
                path = generated / target_id / f"{condition}.png"
                record = render_native_reference(path, select_native_strokes(strokes))
                record.update({"reference_id": "page-004-native", "kind": "full-page"})
                reference_records.append(record)
            else:
                for reference in references:
                    native_bbox = _map_bbox(reference["bbox"], source_size)
                    selected = select_native_strokes(strokes, native_bbox)
                    path = generated / target_id / f"{condition}-{reference['reference_id']}.png"
                    record = render_native_reference(path, selected, native_bbox=native_bbox)
                    record.update(
                        {
                            "reference_id": reference["reference_id"],
                            "kind": "line",
                            "source_bbox": reference["source_bbox"],
                            "source_artifact": reference["source_artifact"],
                            "native_bbox": native_bbox,
                        }
                    )
                    reference_records.append(record)
            images = [Image.open(record["path"]).convert("RGB") for record in reference_records]
            images.append(Image.open(target["path"]).convert("RGB"))
            observed: Any = None

            def observe(value: dict[str, Any]) -> None:
                nonlocal observed
                observed = value

            reader = (
                reader_factory(observe)
                if reader_factory
                else OllamaExemplarReader(observer=observe)
            )
            prompt = ISOLATED_PROMPT if condition == "baseline" else NATIVE_PROMPT
            try:
                analysis = _condition_analysis(
                    _read_configuration(reader, images, prompt, controls, runs)
                )
                _add_truncated_evidence(analysis)
                analysis["parsed_responses"] = [
                    {
                        "run": item["run"],
                        "status": item["status"],
                        "reading": item.get("reading"),
                        "thinking": (
                            item.get("raw_response", {})
                            .get("message", {})
                            .get("thinking")
                            if isinstance(item.get("raw_response"), dict)
                            and isinstance(item["raw_response"].get("message"), dict)
                            else None
                        ),
                        "truncated_evidence": item.get("truncated_evidence"),
                    }
                    for item in analysis["runs"]
                ]
                model = reader.model
            finally:
                reader.release()
                for image in images:
                    image.close()
            analysis["evaluation"] = {
                "status": "not_evaluated",
                "reason": "no_confirmed_review",
            }
            if target_id in review:
                analysis["evaluation"] = {
                    "metrics": [
                        evaluate_reading(reading, review[target_id])
                        for reading in analysis["readings"]
                    ],
                    "ground_truth": review[target_id],
                }
            artifact["results"].append(
                {
                    "target": target_record,
                    "condition": condition,
                    "reference_condition": condition,
                    "reference_images": reference_records,
                    "reference_stroke_ids": [
                        stroke_id
                        for record in reference_records
                        for stroke_id in record["stroke_ids"]
                    ],
                    "prompt_variant": (
                        "baseline" if condition == "baseline" else "native-style-exemplar"
                    ),
                    "prompt": prompt,
                    "model": model,
                    "request_controls": controls,
                    "image_order": [record.get("reference_id") for record in reference_records]
                    + [target_id],
                    "raw_response_observed": observed is not None,
                    "analysis": analysis,
                }
            )
            artifact["completed_results"].append(f"{target_id}:{condition}")
            checkpoint()
    artifact["status"] = "complete"
    checkpoint()
    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def format_boox_recognition(result: dict[str, Any]) -> str:
    return (
        f"experiment: {result['experiment']}\n"
        f"targets: {len(result['targets'])}\n"
        f"conditions: {len(result['results'])}\n"
        f"output: {result['output']}"
    )

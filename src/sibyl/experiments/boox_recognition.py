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
BOOX_RECOGNITION_NUM_CTX = 8192
BOOX_RECOVERY_NUM_PREDICT = 2048
BOOX_RECOVERY_STROKE_WIDTH = 6
BOOX_RECOVERY_REFERENCE_HEIGHT = 64
BOOX_RECOVERY_BASELINE_ATTEMPTS = 2
BOOX_RECOVERY_TARGETED_REREADS = 1
BOOX_RECOVERY_NATIVE_REFERENCE_SIZES = (1, None)
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
    stroke_width: int = 2,
    presentation_height: int | None = None,
) -> dict[str, Any]:
    """Render native coordinates with fixed geometry and record exact provenance."""
    if stroke_width <= 0:
        raise ValueError("stroke_width must be positive")
    if presentation_height is not None and presentation_height <= 0:
        raise ValueError("presentation_height must be positive")
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
            draw.line(pixels, fill=(0, 0, 0), width=stroke_width, joint="curve")
        elif pixels:
            x, y = pixels[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 0, 0))
    native_dimensions = {"width": image.width, "height": image.height}
    presentation_scale = 1.0
    if presentation_height is not None and image.height != presentation_height:
        presentation_scale = presentation_height / image.height
        presentation_size = (
            max(1, round(image.width * presentation_scale)),
            presentation_height,
        )
        image = image.resize(presentation_size, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "dimensions": {"width": image.width, "height": image.height},
        "native_dimensions": native_dimensions,
        "native_origin": {"x": origin[0], "y": origin[1]},
        "stroke_ids": [stroke.get("stroke_id") for stroke in strokes],
        "point_counts": [
            int(stroke.get("point_count", len(_stroke_points(stroke)))) for stroke in strokes
        ],
        "rendering": {
            "background": "white",
            "stroke_width": stroke_width,
            "coordinate_transform": "identity",
            "crop": "native_bbox" if native_bbox is not None else "full_page",
            "presentation_height": presentation_height,
            "presentation_scale": presentation_scale,
            "resampling": "lanczos" if presentation_height is not None else None,
        },
    }


def _reference_lines(
    catalog: list[dict[str, Any]], selection: str | None
) -> list[dict[str, Any]]:
    if selection is None:
        return catalog
    requested = [item.strip() for item in selection.split(",") if item.strip()]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("--reference-lines must contain unique line IDs")
    known = {line["reference_id"] for line in catalog}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown reference line IDs: {', '.join(unknown)}")
    requested_set = set(requested)
    return [line for line in catalog if line["reference_id"] in requested_set]


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
                        "region_id": region.get("region_id"),
                        "source_bbox": line.get("source_bbox"),
                        "bbox": bbox,
                        "source_artifact": str(path),
                    }
                )
    return sorted(lines, key=lambda item: item["reference_id"])


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label} artifact: {path}")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.is_file():
        return path
    return base / path


def _source_matches(artifact: dict[str, Any], image_path: Path) -> bool:
    source = artifact.get("source")
    return isinstance(source, str) and Path(source).resolve() == image_path.resolve()


def _bbox_overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    try:
        left = max(float(first["left"]), float(second["left"]))
        top = max(float(first["top"]), float(second["top"]))
        right = min(float(first["right"]), float(second["right"]))
        bottom = min(float(first["bottom"]), float(second["bottom"]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area = max(0.0, float(first["right"]) - float(first["left"])) * max(
            0.0, float(first["bottom"]) - float(first["top"])
        )
    except (KeyError, TypeError, ValueError):
        return 0.0
    return intersection / area if area else 0.0


def _canonical_figure_regions(image_path: Path) -> list[dict[str, Any]]:
    """Read existing canonical figure classification without invoking inference."""
    transform_path = image_path.parent / f"{image_path.stem}.sibyl" / "transform.json"
    if not transform_path.is_file():
        return []
    try:
        artifact = json.loads(transform_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(artifact, dict):
        return []
    source = artifact.get("source")
    source_image = source.get("image") if isinstance(source, dict) else source
    if not isinstance(source_image, str) or Path(source_image).resolve() != image_path.resolve():
        return []
    figures: list[dict[str, Any]] = []
    for region in artifact.get("regions", []):
        if not isinstance(region, dict) or region.get("kind") != "figure":
            continue
        source = region.get("source")
        bounds = region.get("bounds")
        if not isinstance(bounds, dict) and isinstance(source, dict):
            bounds = source.get("bounds")
        if isinstance(bounds, dict):
            figures.append({"bounds": bounds, "region": region})
    return figures


def _coarse_recovery_targets(
    compare_path: Path, reread_path: Path, image_path: Path
) -> list[dict[str, Any]]:
    """Build document blocks from accepted coarse crops, retaining line evidence."""
    compare = _read_json(compare_path, "trocr_compare")
    reread = _read_json(reread_path, "transcription_reread")
    if not _source_matches(compare, image_path):
        raise ValueError("trocr-compare artifact source does not match IMAGE")
    if not _source_matches(reread, image_path):
        raise ValueError("transcription-reread artifact source does not match IMAGE")
    reread_regions = {
        region.get("region_id"): region
        for region in reread.get("regions", [])
        if isinstance(region, dict) and isinstance(region.get("region_id"), str)
    }
    figures = _canonical_figure_regions(image_path)
    targets: list[dict[str, Any]] = []
    for region in compare.get("regions", []):
        if not isinstance(region, dict) or not isinstance(region.get("region_id"), str):
            continue
        crop = region.get("crop")
        if not isinstance(crop, dict) or not isinstance(crop.get("path"), str):
            continue
        source_bbox = crop.get("source_bbox")
        if isinstance(source_bbox, dict) and any(
            _bbox_overlap_ratio(source_bbox, figure["bounds"]) >= 0.5 for figure in figures
        ):
            continue
        region_id = region["region_id"]
        reread_region = reread_regions.get(region_id, {})
        line_localization = reread_region.get("line_localization", {})
        line_regions = line_localization.get("regions", [])
        line_evidence = {
            "parent_region_id": region_id,
            "source_artifact": str(reread_path),
            "status": line_localization.get("status"),
            "error": line_localization.get("error"),
            "raw_response": line_localization.get("raw_response"),
            "rejected_regions": line_localization.get("rejected_regions", []),
            "regions": line_regions if isinstance(line_regions, list) else [],
        }
        targets.append(
            {
                "target_id": region_id,
                "kind": "region",
                "path": _resolve_path(crop["path"], compare_path.parent.parent),
                "source_bbox": source_bbox,
                "source_coordinate_space": crop.get("source_coordinate_space"),
                "source_artifact": str(compare_path),
                "metadata": crop,
                "line_evidence": line_evidence,
            }
        )
    targets.sort(
        key=lambda target: (
            (target.get("source_bbox") or {}).get("top", 0),
            (target.get("source_bbox") or {}).get("left", 0),
            target["target_id"],
        )
    )
    if not targets:
        raise ValueError("no accepted coarse text-region targets were found")
    return targets


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


def _line_region(line: dict[str, Any]) -> str | None:
    value = line.get("region_id")
    if isinstance(value, str):
        return value
    identifier = line.get("reference_id") or line.get("target_id")
    if isinstance(identifier, str) and "-line-" in identifier:
        return identifier.split("-line-", 1)[0]
    return None


def _select_recovery_reading(analysis: dict[str, Any]) -> dict[str, Any]:
    """Select only defensible repeated evidence; never use semantic plausibility."""
    readings = [item for item in analysis["readings"] if isinstance(item, str)]
    if len(readings) < 2:
        return {
            "status": "unresolved",
            "reading": None,
            "rule": "requires_multiple_successful_runs",
            "reason": "fewer_than_two_successful_runs",
            "candidates": readings,
        }
    if len(set(readings)) == 1:
        return {
            "status": "selected",
            "reading": readings[0],
            "rule": "exact_agreement",
            "reason": "all_successful_runs_agree",
            "candidates": readings,
        }
    normalized: dict[str, list[str]] = {}
    for reading in readings:
        key = " ".join(reading.casefold().split())
        normalized.setdefault(key, []).append(reading)
    ordered = sorted(normalized.items(), key=lambda item: (-len(item[1]), item[0]))
    if ordered and len(ordered[0][1]) > len(readings) / 2:
        return {
            "status": "selected",
            "reading": ordered[0][1][0],
            "rule": "normalized_majority",
            "reason": "strict_normalized_majority",
            "candidates": readings,
        }
    return {
        "status": "unresolved",
        "reading": None,
        "rule": "no_majority",
        "reason": "successful_runs_disagree",
        "candidates": readings,
    }


def _analysis_from_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    readings = [item["reading"] for item in runs if isinstance(item.get("reading"), str)]
    failures = [item for item in runs if item.get("status") != "ok"]
    return _condition_analysis(
        {
            "runs": runs,
            "readings": readings,
            "normalized_readings": [" ".join(item.casefold().split()) for item in readings],
            "distinct_readings": list(dict.fromkeys(readings)),
            "failures": failures,
            "invalid_count": sum(item.get("status") == "invalid_response" for item in runs),
        }
    )


def _adaptive_recovery(
    read_stage: Callable[[str, int | None, int], list[dict[str, Any]]],
    *,
    baseline_attempts: int,
    targeted_rereads: int,
    native_reference_sizes: tuple[int | None, ...],
    native_enabled: bool,
    native_runs: int,
) -> dict[str, Any]:
    """Run recovery stages until the unchanged conservative selector accepts."""
    if baseline_attempts <= 0 or targeted_rereads < 0 or native_runs <= 0:
        raise ValueError("recovery budgets must be positive where applicable")
    all_baseline: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    def attempt(
        stage: str, condition: str, reference_size: int | None, count: int
    ) -> dict[str, Any]:
        runs = read_stage(condition, reference_size, count)
        stages.append(
            {
                "stage": stage,
                "condition": condition,
                "runs": runs,
                "calls": len(runs),
                "selection": None,
            }
        )
        return stages[-1]

    def finish(status: str, selection: dict[str, Any], reason: str) -> dict[str, Any]:
        for stage in stages:
            stage["continue_reason"] = (
                "selected"
                if stage["selection"]["status"] == "selected"
                else stage["selection"]["reason"]
            )
        all_runs = [run for stage in stages for run in stage["runs"]]
        return {
            "status": status,
            "selection": selection,
            "stages": stages,
            "stop_reason": reason,
            "total_model_calls": len(all_runs),
            "successful_reads": [
                run["reading"]
                for run in all_runs
                if isinstance(run.get("reading"), str)
            ],
            "failures": [run for run in all_runs if run.get("status") != "ok"],
        }

    for attempt_number in range(1, baseline_attempts + 1):
        stage = attempt(
            "baseline" if attempt_number == 1 else "baseline-retry",
            "baseline",
            None,
            1,
        )
        all_baseline.extend(stage["runs"])
        selection = _select_recovery_reading(_analysis_from_runs(all_baseline))
        stage["selection"] = selection
        if selection["status"] == "selected":
            return finish("selected", selection, "stop_on_baseline_selection")

    for _ in range(1, targeted_rereads + 1):
        stage = attempt("targeted-baseline-reread", "baseline", None, 1)
        all_baseline.extend(stage["runs"])
        selection = _select_recovery_reading(_analysis_from_runs(all_baseline))
        stage["selection"] = selection
        if selection["status"] == "selected":
            return finish("selected", selection, "stop_on_targeted_reread_selection")

    if native_enabled:
        for index, size in enumerate(native_reference_sizes, start=1):
            condition = "native-exemplar" if index == 1 else "multi-exemplar"
            stage = attempt(
                "native-reference" if index == 1 else "native-reference-expansion",
                condition,
                size,
                native_runs,
            )
            selection = _select_recovery_reading(_analysis_from_runs(stage["runs"]))
            stage["selection"] = selection
            if selection["status"] == "selected":
                return finish("selected", selection, "stop_on_native_selection")
    return finish(
        "unresolved",
        stages[-1]["selection"]
        if stages
        else _select_recovery_reading(_analysis_from_runs(all_baseline)),
        "all_recovery_stages_exhausted",
    )


def _write_page_recovery(
    artifact: dict[str, Any], image_path: Path
) -> tuple[Path, Path]:
    output_directory = image_path.parent / f"{image_path.stem}.sibyl"
    output_directory.mkdir(parents=True, exist_ok=True)
    selected_results: dict[str, dict[str, Any]] = {}
    adaptive_results: dict[str, dict[str, Any]] = {}
    for result in artifact["results"]:
        if result["condition"] == "leave-one-region-out":
            selected_results[result["target"]["target_id"]] = _select_recovery_reading(
                result["analysis"]
            )
    for target in artifact.get("targets", []):
        adaptive = target.get("adaptive_recovery")
        if isinstance(adaptive, dict) and isinstance(adaptive.get("selection"), dict):
            adaptive_results[target["target_id"]] = adaptive["selection"]
    markdown_lines: list[str] = []
    recovery_targets: list[dict[str, Any]] = []
    for target in artifact["targets"]:
        target_id = target["target_id"]
        selection = adaptive_results.get(target_id, selected_results.get(
            target_id,
            {
                "status": "unresolved",
                "reading": None,
                "rule": "missing_recovery_condition",
                "reason": "leave_one_region_out_result_missing",
                "candidates": [],
            },
        ))
        markdown_lines.append(
            selection["reading"] if selection["reading"] is not None else "⟦unresolved⟧"
        )
        recovery_targets.append({"target": target, "selection": selection})
    recovery = {
        "status": "complete",
        "source": artifact["source"],
        "source_sha256": artifact["source_sha256"],
        "note": artifact["native_source"],
        "targets": recovery_targets,
        "selection_condition": "adaptive" if adaptive_results else "leave-one-region-out",
        "summary": artifact.get("recovery_summary"),
        "markdown": str(output_directory / "recovery.md"),
        "evidence": str(output_directory / "recovery.json"),
    }
    artifact["recovery"] = recovery
    evidence_path = output_directory / "recovery.json"
    evidence_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = output_directory / "recovery.md"
    markdown_path.write_text("\n\n".join(markdown_lines) + "\n", encoding="utf-8")
    return markdown_path, evidence_path


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
    num_predict: int | None = None,
    num_ctx: int = BOOX_RECOGNITION_NUM_CTX,
    native_stroke_width: int | None = None,
    reference_height: int | None = None,
    reference_lines: str | None = None,
    conditions: str | None = None,
    baseline_attempts: int = BOOX_RECOVERY_BASELINE_ATTEMPTS,
    targeted_rereads: int = BOOX_RECOVERY_TARGETED_REREADS,
    native_escalation: bool = True,
    native_reference_sizes: str | None = None,
    review_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    reread_path: Path = DEFAULT_REREAD,
    compare_path: Path = DEFAULT_COMPARE,
    reader_factory: ReaderFactory | None = None,
    markdown: bool = False,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    if num_ctx <= 0:
        raise ValueError("num_ctx must be positive")
    if baseline_attempts <= 0 or targeted_rereads < 0:
        raise ValueError("recovery budgets must be positive where applicable")
    if markdown and conditions is None:
        conditions = "leave-one-region-out"
    requested_conditions = selected_conditions(conditions)
    if markdown:
        if num_predict is None:
            num_predict = BOOX_RECOVERY_NUM_PREDICT
        if native_stroke_width is None:
            native_stroke_width = BOOX_RECOVERY_STROKE_WIDTH
        if reference_height is None:
            reference_height = BOOX_RECOVERY_REFERENCE_HEIGHT
    if num_predict is None:
        num_predict = BOOX_RECOGNITION_NUM_PREDICT
    if native_stroke_width is None:
        native_stroke_width = 2
    if num_predict <= 0:
        raise ValueError("num_predict must be positive")
    if native_stroke_width <= 0:
        raise ValueError("native_stroke_width must be positive")
    if reference_height is not None and reference_height <= 0:
        raise ValueError("reference_height must be positive")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not note_path.is_file():
        raise FileNotFoundError(f"Source file not found: {note_path}")
    catalog = _line_catalog(reread_path, image_path)
    if lines is None and regions is None:
        if markdown:
            targets = _coarse_recovery_targets(compare_path, reread_path, image_path)
        else:
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
    if markdown:
        target_order = {
            target["target_id"]: index for index, target in enumerate(targets)
        }
        targets.sort(key=lambda target: target_order.get(target["target_id"], len(catalog)))
    native = _verified_page(note_path)
    source_size = (Image.open(image_path).width, Image.open(image_path).height)
    selected_reference_lines = _reference_lines(catalog, reference_lines)
    if native_reference_sizes is None:
        configured_native_sizes: tuple[int | None, ...] = BOOX_RECOVERY_NATIVE_REFERENCE_SIZES
    else:
        configured_native_sizes_list: list[int | None] = []
        for raw_size in native_reference_sizes.split(","):
            value = raw_size.strip().casefold()
            if value == "all":
                configured_native_sizes_list.append(None)
            elif value:
                try:
                    parsed_size = int(value)
                except ValueError as error:
                    raise ValueError(
                        "native reference sizes must be positive integers or all"
                    ) from error
                if parsed_size <= 0:
                    raise ValueError("native reference sizes must be positive")
                configured_native_sizes_list.append(parsed_size)
        if not configured_native_sizes_list:
            raise ValueError("native reference sizes must not be empty")
        configured_native_sizes = tuple(dict.fromkeys(configured_native_sizes_list))
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
        "num_ctx": num_ctx,
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
        "reference_render_controls": {
            "native_stroke_width": native_stroke_width,
            "reference_height": reference_height,
            "reference_lines": [line["reference_id"] for line in selected_reference_lines],
        },
        "targets": [],
        "evaluation_review": str(review_path) if review_path else None,
        "results": [],
        "output": str(output_path),
        "recovery_requested": markdown,
    }
    if markdown:
        artifact["recovery_controls"] = {
            "baseline_attempts": baseline_attempts,
            "targeted_rereads": targeted_rereads,
            "native_escalation": native_escalation,
            "native_reference_sizes": [
                size if size is not None else "all" for size in configured_native_sizes
            ],
        }
    checkpoint()
    strokes = native["strokes"]
    if markdown:
        page_summary = {
            "total_targets": len(targets),
            "resolved_baseline": 0,
            "resolved_retry": 0,
            "resolved_native": 0,
            "unresolved": 0,
            "model_calls": 0,
        }
        for target in targets:
            target_id = target["target_id"]
            target_bbox = _bbox_tuple(target.get("source_bbox"))
            target_record = {
                "target_id": target_id,
                "kind": target["kind"],
                "path": str(target["path"]),
                "sha256": _sha256(Path(target["path"])),
                "source_bbox": target.get("source_bbox"),
                "native_bbox": _map_bbox(target_bbox, source_size) if target_bbox else None,
                "line_evidence": target.get("line_evidence", {}),
            }
            artifact["targets"].append(target_record)
            other_lines = [
                line
                for line in selected_reference_lines
                if target_bbox is None or not _intersects(target_bbox, line["bbox"])
            ]
            if target["kind"] == "region":
                other_lines = [line for line in other_lines if line.get("region_id") != target_id]
                reference_selection = {
                    "policy": "other_coarse_regions_excluding_target",
                    "target_region": target_id,
                    "eligible_reference_lines": [line["reference_id"] for line in other_lines],
                    "reason": (
                        "other_region_references_selected"
                        if other_lines
                        else "insufficient_other_region_references"
                    ),
                }
            else:
                target_region = _line_region({"target_id": target_id})
                other_lines = [line for line in other_lines if _line_region(line) == target_region]
                reference_selection = {
                    "policy": "same_region_excluding_target",
                    "target_region": target_region,
                    "eligible_reference_lines": [line["reference_id"] for line in other_lines],
                    "reason": (
                        "same_region_references_selected"
                        if other_lines
                        else "insufficient_same_region_references"
                    ),
                }

            def read_stage(
                condition: str,
                reference_size: int | None,
                stage_runs: int,
                *,
                target: dict[str, Any] = target,
                target_id: str = target_id,
                target_record: dict[str, Any] = target_record,
                other_lines: list[dict[str, Any]] = other_lines,
                reference_selection: dict[str, Any] = reference_selection,
            ) -> list[dict[str, Any]]:
                references = [] if condition == "baseline" else (
                    other_lines if reference_size is None else other_lines[:reference_size]
                )
                reference_records: list[dict[str, Any]] = []
                if condition != "baseline":
                    for reference in references:
                        native_bbox = _map_bbox(reference["bbox"], source_size)
                        path = generated / target_id / (
                            f"adaptive-{condition}-{reference['reference_id']}.png"
                        )
                        record = render_native_reference(
                            path,
                            select_native_strokes(strokes, native_bbox),
                            native_bbox=native_bbox,
                            stroke_width=native_stroke_width,
                            presentation_height=reference_height,
                        )
                        record.update({
                            "reference_id": reference["reference_id"],
                            "kind": "line",
                            "source_bbox": reference["source_bbox"],
                            "source_artifact": reference["source_artifact"],
                            "native_bbox": native_bbox,
                        })
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
                        _read_configuration(reader, images, prompt, controls, stage_runs)
                    )
                    _add_truncated_evidence(analysis)
                    analysis["parsed_responses"] = [
                        {
                            "run": item["run"],
                            "status": item["status"],
                            "reading": item.get("reading"),
                            "thinking": item.get("raw_response", {})
                            .get("message", {})
                            .get("thinking")
                            if isinstance(item.get("raw_response"), dict)
                            and isinstance(item["raw_response"].get("message"), dict)
                            else None,
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
                result = {
                    "target": target_record,
                    "condition": condition,
                    "reference_condition": condition,
                    "reference_images": reference_records,
                    "reference_stroke_ids": [
                        stroke_id
                        for record in reference_records
                        for stroke_id in record["stroke_ids"]
                    ],
                    "reference_render_controls": {
                        "native_stroke_width": native_stroke_width,
                        "reference_height": reference_height,
                        "reference_lines": [reference["reference_id"] for reference in references],
                    },
                    "reference_selection": reference_selection,
                    "prompt_variant": (
                        "baseline"
                        if condition == "baseline"
                        else "native-style-exemplar"
                    ),
                    "prompt": prompt,
                    "model": model,
                    "request_controls": controls,
                    "image_order": [
                        record.get("reference_id") for record in reference_records
                    ]
                    + [target_id],
                    "raw_response_observed": observed is not None,
                    "analysis": analysis,
                }
                artifact["results"].append(result)
                artifact["completed_results"].append(f"{target_id}:{condition}")
                return cast(list[dict[str, Any]], analysis["runs"])

            result_start = len(artifact["results"])
            adaptive = _adaptive_recovery(
                read_stage,
                baseline_attempts=baseline_attempts,
                targeted_rereads=targeted_rereads,
                native_reference_sizes=configured_native_sizes,
                native_enabled=native_escalation,
                native_runs=runs,
            )
            for result, stage in zip(
                artifact["results"][result_start:], adaptive["stages"], strict=True
            ):
                result["stage"] = stage["stage"]
            target_record["adaptive_recovery"] = adaptive
            page_summary["model_calls"] += sum(stage["calls"] for stage in adaptive["stages"])
            if adaptive["status"] == "selected":
                selected_stage = adaptive["stages"][-1]["stage"]
                if selected_stage == "baseline":
                    page_summary["resolved_baseline"] += 1
                elif selected_stage in {"baseline-retry", "targeted-baseline-reread"}:
                    page_summary["resolved_retry"] += 1
                else:
                    page_summary["resolved_native"] += 1
            else:
                page_summary["unresolved"] += 1
            checkpoint()
        artifact["recovery_summary"] = page_summary
        artifact["status"] = "complete"
        _write_page_recovery(artifact, image_path)
        checkpoint()
        return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))
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
        if markdown:
            target_record["line_evidence"] = target.get("line_evidence", {
                "parent_region_id": target_id,
                "source_artifact": str(reread_path),
                "status": "not_available",
                "error": "no line-localization record for coarse region",
                "raw_response": None,
                "rejected_regions": [],
                "regions": [],
            })
        artifact["targets"].append(target_record)
        checkpoint()
        other_lines = [
            line
            for line in selected_reference_lines
            if target_bbox is None
            or not _intersects(target_bbox, line["bbox"])
            or (line["reference_id"] == target_id and target["kind"] != "line")
        ]
        other_lines = [line for line in other_lines if line["reference_id"] != target_id]
        if markdown:
            if target["kind"] == "region":
                other_lines = [
                    line for line in other_lines if line.get("region_id") != target_id
                ]
                reference_selection = {
                    "policy": "other_coarse_regions_excluding_target",
                    "target_region": target_id,
                    "eligible_reference_lines": [
                        line["reference_id"] for line in other_lines
                    ],
                    "reason": (
                        "other_region_references_selected"
                        if other_lines
                        else "insufficient_other_region_references"
                    ),
                }
            else:
                target_region = _line_region({"target_id": target_id})
                other_lines = [
                    line for line in other_lines if _line_region(line) == target_region
                ]
                reference_selection = {
                    "policy": "same_region_excluding_target",
                    "target_region": target_region,
                    "eligible_reference_lines": [line["reference_id"] for line in other_lines],
                    "reason": (
                        "same_region_references_selected"
                        if other_lines
                        else "insufficient_same_region_references"
                    ),
                }
        else:
            reference_selection = {
                "policy": "configured_experiment_references",
                "eligible_reference_lines": [line["reference_id"] for line in other_lines],
                "reason": "target_excluded_structurally",
            }
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
                record = render_native_reference(
                    path,
                    select_native_strokes(strokes),
                    stroke_width=native_stroke_width,
                    presentation_height=reference_height,
                )
                record.update({"reference_id": "page-004-native", "kind": "full-page"})
                reference_records.append(record)
            else:
                for reference in references:
                    native_bbox = _map_bbox(reference["bbox"], source_size)
                    selected = select_native_strokes(strokes, native_bbox)
                    path = generated / target_id / f"{condition}-{reference['reference_id']}.png"
                    record = render_native_reference(
                        path,
                        selected,
                        native_bbox=native_bbox,
                        stroke_width=native_stroke_width,
                        presentation_height=reference_height,
                    )
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
                    "reference_render_controls": {
                        "native_stroke_width": native_stroke_width,
                        "reference_height": reference_height,
                        "reference_lines": [
                            reference["reference_id"] for reference in references
                        ],
                    },
                    "reference_selection": reference_selection,
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
    if markdown:
        _write_page_recovery(artifact, image_path)
    checkpoint()
    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def format_boox_recognition(result: dict[str, Any]) -> str:
    lines = [
        f"experiment: {result['experiment']}\n"
        f"targets: {len(result['targets'])}\n"
        f"conditions: {len(result['results'])}\n"
        f"output: {result['output']}"
    ]
    recovery = result.get("recovery")
    if isinstance(recovery, dict):
        lines.extend([f"recovery: {recovery['markdown']}", f"evidence: {recovery['evidence']}"])
    summary = result.get("recovery_summary")
    if isinstance(summary, dict):
        lines.extend(
            [
                f"resolved baseline: {summary['resolved_baseline']}",
                f"resolved retry: {summary['resolved_retry']}",
                f"resolved native: {summary['resolved_native']}",
                f"unresolved: {summary['unresolved']}",
                f"model calls: {summary['model_calls']}",
            ]
        )
    return "\n".join(lines)

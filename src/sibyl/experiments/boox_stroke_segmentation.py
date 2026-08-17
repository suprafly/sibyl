"""Experiment comparing image segmentation strategies using native BOOX geometry."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

from sibyl.experiments.boox_recognition import (
    _canonical_figure_regions,
    _coarse_recovery_targets,
    _line_catalog,
    _select_recovery_reading,
)
from sibyl.experiments.boox_strokes import inspect_boox_strokes
from sibyl.experiments.handwriting_exemplars import (
    ExemplarReader,
    OllamaExemplarReader,
    _read_configuration,
)
from sibyl.experiments.qwen_recognition_knobs import ISOLATED_PROMPT
from sibyl.transform import DEFAULT_QWEN_MODEL

DEFAULT_IMAGE = Path("samples/Grafting-101-page-004.png")
DEFAULT_NOTE = Path("samples/Grafting 101.note")
DEFAULT_OUTPUT = Path(".sibyl/experiments/stroke-segmentation.json")
DEFAULT_RUNS = 1
PAGE = 4
NATIVE_SIZE = (1404, 1872)
ASPECT_RATIO_TOLERANCE = 0.005
DEFAULT_VERTICAL_GAP = 70.0
DEFAULT_WORD_GAP = 55.0
DEFAULT_MIN_STROKES = 2
DEFAULT_MIN_WIDTH = 8.0
DEFAULT_MIN_HEIGHT = 8.0

ReaderFactory = Callable[[Callable[[dict[str, Any]], None]], ExemplarReader]


def _native_raster_mapping(
    raster_size: tuple[int, int],
    *,
    aspect_ratio_tolerance: float = ASPECT_RATIO_TOLERANCE,
) -> dict[str, Any]:
    """Return the deterministic page-coordinate mapping for this raster."""
    raster_width, raster_height = raster_size
    native_width, native_height = NATIVE_SIZE
    if raster_width <= 0 or raster_height <= 0:
        raise ValueError("raster page dimensions must be positive")
    if aspect_ratio_tolerance < 0:
        raise ValueError("aspect-ratio tolerance must be non-negative")
    native_ratio = native_width / native_height
    raster_ratio = raster_width / raster_height
    relative_error = abs(raster_ratio - native_ratio) / native_ratio
    if relative_error > aspect_ratio_tolerance:
        raise ValueError(
            "raster page aspect ratio is incompatible with the verified BOOX page"
        )
    scale_x = raster_width / native_width
    scale_y = raster_height / native_height
    return {
        "native_dimensions": {"width": native_width, "height": native_height},
        "raster_dimensions": {"width": raster_width, "height": raster_height},
        "scale_x": scale_x,
        "scale_y": scale_y,
        "aspect_ratio_tolerance": aspect_ratio_tolerance,
        "aspect_ratio_relative_error": relative_error,
        "uniform_scaling": scale_x == scale_y,
        "coordinate_transform": "native-to-raster",
    }


def _map_bbox(
    native_bbox: dict[str, float], mapping: dict[str, Any]
) -> dict[str, float]:
    """Map a native bbox into raster coordinates without changing its geometry."""
    scale_x = float(mapping["scale_x"])
    scale_y = float(mapping["scale_y"])
    return {
        "left": native_bbox["left"] * scale_x,
        "top": native_bbox["top"] * scale_y,
        "right": native_bbox["right"] * scale_x,
        "bottom": native_bbox["bottom"] * scale_y,
    }


def _unmap_bbox(raster_bbox: dict[str, float], mapping: dict[str, Any]) -> dict[str, float]:
    inverse = {
        "scale_x": 1 / float(mapping["scale_x"]),
        "scale_y": 1 / float(mapping["scale_y"]),
    }
    return _map_bbox(raster_bbox, inverse)


def _map_group(group: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    native_bbox = cast(dict[str, float], group["native_bbox"])
    raster_bbox = _map_bbox(native_bbox, mapping)
    return {
        **group,
        "native_bbox": native_bbox.copy(),
        "raster_bbox": raster_bbox,
        "source_bbox": raster_bbox.copy(),
    }


def _map_rejected(item: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    native_bbox = item.get("bbox")
    if not isinstance(native_bbox, dict):
        return item
    raster_bbox = _map_bbox(cast(dict[str, float], native_bbox), mapping)
    return {
        **item,
        "native_bbox": dict(native_bbox),
        "raster_bbox": raster_bbox,
        "bbox": raster_bbox,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stroke_id(stroke: dict[str, Any]) -> str:
    for key in ("stroke_id", "shape_id", "point_resource_id"):
        value = stroke.get(key)
        if isinstance(value, str) and value:
            return value
    return f"stroke-{int(stroke.get('order', 0)) + 1:03d}"


def _bbox(stroke: dict[str, Any]) -> dict[str, float] | None:
    value = stroke.get("native_bounds") or stroke.get("bounds")
    if not isinstance(value, dict):
        return None
    try:
        left, top, right, bottom = (float(value[key]) for key in ("left", "top", "right", "bottom"))
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _union_bbox(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "left": min(item["left"] for item in items),
        "top": min(item["top"] for item in items),
        "right": max(item["right"] for item in items),
        "bottom": max(item["bottom"] for item in items),
    }


def _vertical_gap(first: dict[str, float], second: dict[str, float]) -> float:
    if first["bottom"] >= second["top"] and second["bottom"] >= first["top"]:
        return 0.0
    return max(first["top"] - second["bottom"], second["top"] - first["bottom"])


def _vertical_center(value: dict[str, float]) -> float:
    return (value["top"] + value["bottom"]) / 2


def _horizontal_gap(first: dict[str, float], second: dict[str, float]) -> float:
    if first["right"] >= second["left"] and second["right"] >= first["left"]:
        return 0.0
    return max(first["left"] - second["right"], second["left"] - first["right"])


def _group_record(
    strokes: list[dict[str, Any]],
    *,
    kind: str,
    order: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    boxes = [cast(dict[str, float], stroke["_bbox"]) for stroke in strokes]
    bounds = _union_bbox(boxes)
    return {
        "group_id": f"boox-{kind}-{order + 1:03d}",
        "kind": kind,
        "order": order,
        "stroke_ids": [_stroke_id(stroke) for stroke in strokes],
        "point_count": sum(
            int(stroke.get("point_count", len(stroke.get("native_points", []))))
            for stroke in strokes
        ),
        "native_bbox": bounds,
        "source_bbox": bounds.copy(),
        "stroke_count": len(strokes),
        "grouping_parameters": parameters,
    }


def derive_boox_groups(
    strokes: list[dict[str, Any]],
    *,
    max_vertical_gap: float = DEFAULT_VERTICAL_GAP,
    max_word_gap: float = DEFAULT_WORD_GAP,
    min_strokes: int = DEFAULT_MIN_STROKES,
    min_width: float = DEFAULT_MIN_WIDTH,
    min_height: float = DEFAULT_MIN_HEIGHT,
    figure_regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cluster complete strokes into inspectable line and word candidates."""
    if max_vertical_gap < 0 or max_word_gap < 0 or min_strokes <= 0:
        raise ValueError("grouping distances must be non-negative and min_strokes positive")
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for stroke in sorted(strokes, key=lambda item: (item.get("order", 0), _stroke_id(item))):
        bounds = _bbox(stroke)
        if bounds is None:
            rejected.append({"stroke_ids": [_stroke_id(stroke)], "reason": "missing_or_empty_bbox"})
            continue
        valid.append({**stroke, "_bbox": bounds})
    lines: list[list[dict[str, Any]]] = []
    for stroke in sorted(
        valid, key=lambda item: (item["_bbox"]["top"], item["_bbox"]["left"], item.get("order", 0))
    ):
        matches = [
            group
            for group in lines
            if abs(
                _vertical_center(_union_bbox([item["_bbox"] for item in group]))
                - _vertical_center(stroke["_bbox"])
            )
            <= max_vertical_gap
        ]
        if matches:
            min(
                matches,
                key=lambda group: abs(
                    _vertical_center(_union_bbox([item["_bbox"] for item in group]))
                    - _vertical_center(stroke["_bbox"])
                ),
            ).append(stroke)
        else:
            lines.append([stroke])
    lines.sort(
        key=lambda group: (
            _union_bbox([item["_bbox"] for item in group])["top"],
            _union_bbox([item["_bbox"] for item in group])["left"],
        )
    )
    line_records: list[dict[str, Any]] = []
    word_records: list[dict[str, Any]] = []
    for line_order, line in enumerate(lines):
        line.sort(key=lambda item: (item["_bbox"]["left"], item.get("order", 0)))
        line_record = _group_record(
            line,
            kind="line",
            order=line_order,
            parameters={
                "max_vertical_gap": max_vertical_gap,
                "min_strokes": min_strokes,
                "min_width": min_width,
                "min_height": min_height,
            },
        )
        bounds = line_record["source_bbox"]
        if (
            len(line) < min_strokes
            or bounds["right"] - bounds["left"] < min_width
            or bounds["bottom"] - bounds["top"] < min_height
        ):
            rejected.append(
                {
                    "stroke_ids": line_record["stroke_ids"],
                    "reason": "line_below_minimum",
                    "bbox": bounds,
                }
            )
            continue
        if figure_regions and any(
            _overlap_ratio(bounds, figure.get("bounds", {})) >= 0.5 for figure in figure_regions
        ):
            rejected.append(
                {
                    "stroke_ids": line_record["stroke_ids"],
                    "reason": "overlaps_figure",
                    "bbox": bounds,
                }
            )
            continue
        line_record["group_id"] = f"boox-line-{len(line_records) + 1:03d}"
        line_record["order"] = len(line_records)
        line_records.append(line_record)
        words: list[list[dict[str, Any]]] = []
        for stroke in line:
            if not words or _horizontal_gap(words[-1][-1]["_bbox"], stroke["_bbox"]) > max_word_gap:
                words.append([stroke])
            else:
                words[-1].append(stroke)
        for word_order, word in enumerate(words):
            word_record = _group_record(
                word,
                kind="word",
                order=len(word_records),
                parameters={
                    "max_vertical_gap": max_vertical_gap,
                    "max_word_gap": max_word_gap,
                    "parent_line_id": line_record["group_id"],
                },
            )
            word_record["line_id"] = line_record["group_id"]
            word_record["word_order"] = word_order
            if (
                len(word) >= min_strokes
                and word_record["source_bbox"]["right"] - word_record["source_bbox"]["left"]
                >= min_width
            ):
                word_records.append(word_record)
            else:
                rejected.append(
                    {
                        "stroke_ids": word_record["stroke_ids"],
                        "reason": "word_below_minimum",
                        "bbox": word_record["source_bbox"],
                    }
                )
    return {
        "lines": line_records,
        "words": word_records,
        "rejected": rejected,
        "parameters": {
            "max_vertical_gap": max_vertical_gap,
            "max_word_gap": max_word_gap,
            "min_strokes": min_strokes,
            "min_width": min_width,
            "min_height": min_height,
            "coordinate_transform": "identity",
        },
    }


def _overlap_ratio(first: dict[str, float], second: dict[str, Any]) -> float:
    try:
        left = max(first["left"], float(second["left"]))
        top = max(first["top"], float(second["top"]))
        right = min(first["right"], float(second["right"]))
        bottom = min(first["bottom"], float(second["bottom"]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area = (first["right"] - first["left"]) * (first["bottom"] - first["top"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return intersection / area if area else 0.0


def _crop_group(
    image: Image.Image,
    group: dict[str, Any],
    output_directory: Path,
    *,
    padding: int = 8,
) -> dict[str, Any]:
    bounds = group.get("raster_bbox", group["source_bbox"])
    crop_bbox = {
        "left": max(0, round(bounds["left"] - padding)),
        "top": max(0, round(bounds["top"] - padding)),
        "right": min(image.width, round(bounds["right"] + padding)),
        "bottom": min(image.height, round(bounds["bottom"] + padding)),
    }
    path = output_directory / f"{group['group_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(
        (crop_bbox["left"], crop_bbox["top"], crop_bbox["right"], crop_bbox["bottom"])
    ).convert("RGB").save(path, format="PNG", optimize=False)
    return {
        **group,
        "crop_path": str(path),
        "crop_sha256": _sha256(path),
        "crop_bbox": crop_bbox,
        "padding": padding,
    }


def _existing_crop_target(target: dict[str, Any]) -> dict[str, Any]:
    path = Path(target["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Existing segmentation crop not found: {path}")
    return {
        **target,
        "path": str(path),
        "group_id": target["target_id"],
        "crop_path": str(path),
        "crop_sha256": _sha256(path),
        "crop_bbox": target.get("source_bbox"),
        "padding": target.get("metadata", {}).get("padding"),
    }


def _overlay(
    image_path: Path,
    groups: list[dict[str, Any]],
    path: Path,
    color: tuple[int, int, int],
    rejected: list[dict[str, Any]] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for group in groups:
        bbox = group["source_bbox"]
        coords = (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"])
        draw.rectangle(coords, outline=color, width=3)
        draw.text((bbox["left"], max(0, bbox["top"] - 16)), str(group["order"] + 1), fill=color)
    for item in rejected or []:
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            coords = (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"])
            draw.rectangle(coords, outline=(255, 128, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _normalize(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+|[^\w\s]", _normalize(text), flags=re.UNICODE)


def _edit_distance(first: str, second: str) -> int:
    row = list(range(len(second) + 1))
    for index, left in enumerate(first, start=1):
        next_row = [index]
        for right_index, right in enumerate(second, start=1):
            next_row.append(
                min(next_row[-1] + 1, row[right_index] + 1, row[right_index - 1] + (left != right))
            )
        row = next_row
    return row[-1]


def evaluate_markdown(markdown: str, truth_lines: list[str]) -> dict[str, Any]:
    observed = markdown.splitlines()
    pairs = list(zip(observed, truth_lines, strict=False))
    token_total = sum(len(_tokens(truth)) for truth in truth_lines)
    token_correct = sum(
        sum(token in _tokens(truth) for token in _tokens(observed_line))
        for observed_line, truth in pairs
    )
    return {
        "exact_line_match": sum(observed_line == truth for observed_line, truth in pairs),
        "normalized_line_match": sum(
            _normalize(observed_line) == _normalize(truth) for observed_line, truth in pairs
        ),
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "character_edit_distance": sum(
            _edit_distance(_normalize(observed_line), _normalize(truth))
            for observed_line, truth in pairs
        ),
        "unresolved_count": sum(line.strip() == "⟦unresolved⟧" for line in observed),
        "resolved_block_count": sum(line.strip() != "⟦unresolved⟧" for line in observed),
        "truth_line_count": len(truth_lines),
        "observed_line_count": len(observed),
    }


def _review_lines(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    lines = value.get("lines") if isinstance(value, dict) else value
    if not isinstance(lines, list):
        raise ValueError("review must contain a lines list")
    result: list[str] = []
    for line in lines:
        text = line.get("text") if isinstance(line, dict) else line
        if not isinstance(text, str):
            raise ValueError("review lines must contain text")
        result.append(text)
    return result


def _recognize(
    targets: list[dict[str, Any]],
    *,
    runs: int,
    num_predict: int,
    num_ctx: int,
    reader_factory: ReaderFactory | None,
    prompt: str = ISOLATED_PROMPT,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    markdown: list[str] = []
    controls = {
        "model": DEFAULT_QWEN_MODEL,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "think": False,
        "stream": False,
        "keep_alive": 0,
    }
    for target in targets:
        images = [Image.open(target["crop_path"]).convert("RGB")]
        observed: Any = None

        def observe(value: dict[str, Any]) -> None:
            nonlocal observed
            observed = value

        reader = (
            reader_factory(observe) if reader_factory else OllamaExemplarReader(observer=observe)
        )
        try:
            analysis = _read_configuration(reader, images, prompt, controls, runs)
            selection = _select_recovery_reading({"readings": analysis["readings"]})
            results.append(
                {
                    "target": target,
                    "condition": "image-raster",
                    "prompt": prompt,
                    "request_controls": controls,
                    "model": reader.model,
                    "raw_response_observed": observed is not None,
                    "analysis": analysis,
                    "selection": selection,
                }
            )
            markdown.append(selection["reading"] or "⟦unresolved⟧")
        finally:
            reader.release()
            for image in images:
                image.close()
    return results, markdown


def run_stroke_segmentation(
    image_path: Path = DEFAULT_IMAGE,
    *,
    note_path: Path = DEFAULT_NOTE,
    runs: int = DEFAULT_RUNS,
    num_predict: int = 2048,
    num_ctx: int = 8192,
    review_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    reread_path: Path = Path(".sibyl/experiments/transcription-reread.json"),
    compare_path: Path = Path(".sibyl/experiments/trocr-compare.json"),
    max_vertical_gap: float = DEFAULT_VERTICAL_GAP,
    max_word_gap: float = DEFAULT_WORD_GAP,
    reader_factory: ReaderFactory | None = None,
) -> dict[str, Any]:
    if runs <= 0 or num_predict <= 0 or num_ctx <= 0:
        raise ValueError("runs and decoding controls must be positive")
    if not image_path.is_file() or not note_path.is_file():
        raise FileNotFoundError("page image and BOOX note are required")
    with Image.open(image_path) as source:
        mapping = _native_raster_mapping(source.size)
        source_image = source.convert("RGB")
    native = inspect_boox_strokes(
        note_path, page=PAGE, output=output_path.parent / "stroke-segmentation" / "boox-strokes"
    )
    if (
        len(native["strokes"]) != 167
        or sum(stroke.get("point_count", 0) for stroke in native["strokes"]) != 17272
    ):
        raise ValueError("stroke segmentation requires the verified page-4 decode")
    output_directory = output_path.parent / "stroke-segmentation"
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = _canonical_figure_regions(image_path)
    native_figures = [
        {
            **figure,
            "bounds": _unmap_bbox(cast(dict[str, float], figure["bounds"]), mapping),
        }
        for figure in figures
    ]
    groups = derive_boox_groups(
        native["strokes"],
        max_vertical_gap=max_vertical_gap,
        max_word_gap=max_word_gap,
        figure_regions=native_figures,
    )
    groups["lines"] = [_map_group(group, mapping) for group in groups["lines"]]
    groups["words"] = [_map_group(group, mapping) for group in groups["words"]]
    groups["rejected"] = [_map_rejected(item, mapping) for item in groups["rejected"]]
    groups["parameters"]["coordinate_transform"] = mapping["coordinate_transform"]
    boox_lines = [
        _crop_group(source_image, group, output_directory / "boox-lines")
        for group in groups["lines"]
    ]
    boox_words = [
        _crop_group(source_image, group, output_directory / "boox-words")
        for group in groups["words"]
    ]
    _overlay(
        image_path,
        boox_lines,
        output_directory / "page-004-lines-overlay.png",
        (255, 0, 0),
        groups["rejected"],
    )
    _overlay(
        image_path,
        boox_words,
        output_directory / "page-004-words-overlay.png",
        (0, 0, 255),
        groups["rejected"],
    )
    coarse = _coarse_recovery_targets(compare_path, reread_path, image_path)
    coarse_targets = [_existing_crop_target(target) for target in coarse]
    line_catalog = _line_catalog(reread_path, image_path)
    visual_targets = []
    for line in line_catalog:
        path = (
            Path(line["source_artifact"]).parent
            / "transcription-reread"
            / f"{line['reference_id']}.png"
        )
        if path.is_file():
            visual_targets.append(
                _existing_crop_target({**line, "target_id": line["reference_id"], "path": path})
            )
    review_lines = _review_lines(review_path)
    strategies: dict[str, dict[str, Any]] = {}
    for name, targets in (
        ("coarse", coarse_targets),
        ("visual-lines", visual_targets),
        ("boox-lines", boox_lines),
        ("boox-words", boox_words),
    ):
        results, markdown_lines = _recognize(
            targets,
            runs=runs,
            num_predict=num_predict,
            num_ctx=num_ctx,
            reader_factory=reader_factory,
        )
        markdown_path = output_directory / f"{name}.md"
        markdown_path.write_text("\n\n".join(markdown_lines) + "\n", encoding="utf-8")
        strategy = {
            "target_count": len(targets),
            "targets": targets,
            "results": results,
            "markdown": str(markdown_path),
            "markdown_lines": markdown_lines,
        }
        if review_lines is not None:
            strategy["evaluation"] = evaluate_markdown("\n".join(markdown_lines), review_lines)
        strategies[name] = strategy
    artifact = {
        "experiment": "boox_stroke_segmentation",
        "source": str(image_path),
        "source_sha256": _sha256(image_path),
        "native_source": {
            "note": str(note_path),
            "note_sha256": _sha256(note_path),
            "page": PAGE,
            "page_id": native["selected_page"]["page_id"],
            "dimensions": list(NATIVE_SIZE),
            "stroke_count": len(native["strokes"]),
            "point_count": sum(stroke.get("point_count", 0) for stroke in native["strokes"]),
            "coordinate_transform": mapping["coordinate_transform"],
        },
        "raster_page": mapping,
        "controls": {"runs": runs, "num_predict": num_predict, "num_ctx": num_ctx},
        "grouping": groups,
        "overlays": {
            "lines": str(output_directory / "page-004-lines-overlay.png"),
            "words": str(output_directory / "page-004-words-overlay.png"),
        },
        "review": str(review_path) if review_path else None,
        "strategies": strategies,
        "output": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def format_stroke_segmentation(result: dict[str, Any]) -> str:
    grouping = result["grouping"]
    return (
        f"lines: {len(grouping['lines'])}\n"
        f"words: {len(grouping['words'])}\n"
        f"output: {result['output']}"
    )

"""Controlled preprocessing matrix for preserved handwriting crops."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from sibyl.experiments.transcription_reread import (
    REGIONAL_PROMPT,
    REGIONAL_SCHEMA,
    OllamaRegionalReader,
    RegionalReader,
    requested_runs,
)
from sibyl.experiments.trocr import MODEL_ID as TROCR_MODEL_ID
from sibyl.experiments.trocr import TrocrRecognizer
from sibyl.transform import DEFAULT_QWEN_MODEL

DEFAULT_OUTPUT = Path(".sibyl/experiments/handwriting-preprocess.json")
DEFAULT_COMPARE_ARTIFACT = Path(".sibyl/experiments/trocr-compare.json")
DEFAULT_REREAD_ARTIFACT = Path(".sibyl/experiments/transcription-reread.json")


class TrocrReader(Protocol):
    def recognize(self, image: Image.Image) -> tuple[str, float]: ...


ReaderFactory = Callable[[Callable[[dict[str, Any]], None]], RegionalReader]
TrocrFactory = Callable[[], tuple[TrocrReader, dict[str, Any]]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+|[^\w\s]", _normalize(text), flags=re.UNICODE)


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for left_index, left in enumerate(first, start=1):
        current = [left_index]
        for right_index, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def evaluate_candidate(candidate: str, truth: str) -> dict[str, Any]:
    candidate_tokens = _tokens(candidate)
    truth_tokens = _tokens(truth)
    overlap = sum(token in truth_tokens for token in candidate_tokens)
    return {
        "exact_match": candidate == truth,
        "normalized_exact_match": _normalize(candidate) == _normalize(truth),
        "token_overlap": overlap / len(truth_tokens) if truth_tokens else 0.0,
        "character_edit_distance": _edit_distance(_normalize(candidate), _normalize(truth)),
    }


def selected_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise ValueError("selection must contain at least one identifier")
    return result


def _read_json(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed artifact: {path}") from error
    if not isinstance(value, dict) or value.get("experiment") != expected:
        raise ValueError(f"artifact must be a {expected} artifact")
    return value


def _resolve_path(path: str, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.is_file():
        return candidate
    return base / candidate


def _source_matches(artifact: dict[str, Any], image_path: Path) -> bool:
    source = artifact.get("source")
    if not isinstance(source, str):
        return False
    return Path(source).resolve() == image_path.resolve()


def _targets(
    image_path: Path,
    *,
    regions: str | None,
    lines: str | None,
    crop_path: Path | None,
    compare_path: Path,
    reread_path: Path,
) -> list[dict[str, Any]]:
    if crop_path is not None:
        if not crop_path.is_file():
            raise FileNotFoundError(f"Crop not found: {crop_path}")
        return [
            {
                "target_id": "source-crop",
                "kind": "crop",
                "path": crop_path,
                "source_bbox": None,
                "source_coordinate_space": "provided_crop",
                "source_artifact": None,
                "metadata": {},
            }
        ]

    wanted_lines = selected_ids(lines)
    wanted_regions = selected_ids(regions)
    if wanted_lines is not None:
        reread = _read_json(reread_path, "transcription_reread")
        if not _source_matches(reread, image_path):
            raise ValueError("transcription-reread artifact source does not match IMAGE")
        result: list[dict[str, Any]] = []
        for region in reread.get("regions", []):
            for line in region.get("line_localization", {}).get("regions", []):
                if line.get("line_id") in wanted_lines:
                    result.append(
                        {
                            "target_id": line["line_id"],
                            "kind": "line",
                            "path": _resolve_path(line["path"], reread_path.parent.parent),
                            "source_bbox": line.get("source_bbox"),
                            "source_coordinate_space": line.get("source_coordinate_space"),
                            "source_artifact": str(reread_path),
                            "metadata": line,
                        }
                    )
        found = {item["target_id"] for item in result}
        missing = wanted_lines - found
        if missing:
            raise ValueError(f"unknown line IDs: {', '.join(sorted(missing))}")
        return sorted(result, key=lambda item: item["target_id"])

    artifact = _read_json(compare_path, "trocr_compare")
    if not _source_matches(artifact, image_path):
        raise ValueError("trocr-compare artifact source does not match IMAGE")
    wanted = wanted_regions
    result = []
    for region in artifact.get("regions", []):
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or (wanted is not None and region_id not in wanted):
            continue
        crop = region.get("crop", {})
        result.append(
            {
                "target_id": region_id,
                "kind": "region",
                "path": _resolve_path(str(crop["path"]), compare_path.parent.parent),
                "source_bbox": crop.get("source_bbox"),
                "source_coordinate_space": crop.get("source_coordinate_space"),
                "source_artifact": str(compare_path),
                "metadata": crop,
            }
        )
    if wanted is not None:
        found = {item["target_id"] for item in result}
        missing = wanted - found
        if missing:
            raise ValueError(f"unknown region IDs: {', '.join(sorted(missing))}")
    if not result:
        raise ValueError("no source crop targets were found")
    return result


def _variant_images(source: Image.Image) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    rgb = source.convert("RGB")
    gray = rgb.convert("L")
    normalized = ImageOps.autocontrast(gray)
    return [
        ("rgb-original", rgb.copy(), {"mode": "RGB", "scale": 1}),
        ("grayscale", gray.copy(), {"mode": "L", "scale": 1}),
        (
            "rgb-2x",
            rgb.resize((rgb.width * 2, rgb.height * 2), Image.Resampling.LANCZOS),
            {"mode": "RGB", "scale": 2},
        ),
        (
            "rgb-3x",
            rgb.resize((rgb.width * 3, rgb.height * 3), Image.Resampling.LANCZOS),
            {"mode": "RGB", "scale": 3},
        ),
        (
            "grayscale-2x",
            gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS),
            {"mode": "L", "scale": 2},
        ),
        (
            "contrast-grayscale",
            normalized.copy(),
            {"mode": "L", "scale": 1, "operation": "autocontrast"},
        ),
        (
            "contrast-grayscale-2x",
            normalized.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS),
            {"mode": "L", "scale": 2, "operation": "autocontrast"},
        ),
    ]


def _run_qwen(image: Image.Image, runs: int, factory: ReaderFactory | None) -> dict[str, Any]:
    raw_response: Any = None

    def observe(value: dict[str, Any]) -> None:
        nonlocal raw_response
        raw_response = value

    reader = factory(observe) if factory else OllamaRegionalReader(observer=observe)
    reads: list[dict[str, Any]] = []
    try:
        for number in range(1, runs + 1):
            raw_response = None
            started = time.perf_counter()
            try:
                value, duration_ms = reader.read(image)
                status = value.get("status", "ok")
                raw = raw_response if raw_response is not None else value.get("raw_response")
                if status == "ok" and isinstance(value.get("text"), str):
                    reads.append(
                        {
                            "run": number,
                            "status": "ok",
                            "text": value["text"],
                            "raw_response": raw,
                            "duration_ms": round(duration_ms, 3),
                        }
                    )
                else:
                    reads.append(
                        {
                            "run": number,
                            "status": status or "invalid_response",
                            "text": None,
                            "raw_response": raw,
                            "error": value.get("error", "missing text"),
                            "duration_ms": round(duration_ms, 3),
                        }
                    )
            except (RuntimeError, ValueError) as error:
                reads.append(
                    {
                        "run": number,
                        "status": "request_failure",
                        "text": None,
                        "raw_response": raw_response,
                        "error": str(error),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
    finally:
        reader.release()
    distinct = list(dict.fromkeys(item["text"] for item in reads if item["status"] == "ok"))
    return {
        "model": getattr(reader, "model", DEFAULT_QWEN_MODEL),
        "configuration": {
            "model": getattr(reader, "model", DEFAULT_QWEN_MODEL),
            "think": False,
            "num_predict": 256,
            "stream": False,
            "keep_alive": 0,
            "temperature": "unspecified (Ollama/model default)",
            "top_p": "unspecified (Ollama/model default)",
            "seed": "unspecified (Ollama/model default)",
            "prompt": REGIONAL_PROMPT,
            "schema": REGIONAL_SCHEMA,
        },
        "runs": reads,
        "parsed_readings": [item["text"] for item in reads if isinstance(item.get("text"), str)],
        "normalized_readings": [
            _normalize(item["text"]) for item in reads if isinstance(item.get("text"), str)
        ],
        "distinct_readings": distinct,
        "stable": len(distinct) == 1 and bool(distinct),
        "failures": [item for item in reads if item["status"] != "ok"],
    }


def _run_trocr(
    image: Image.Image, runs: int, recognizer: TrocrReader, metadata: dict[str, Any]
) -> dict[str, Any]:
    reads: list[dict[str, Any]] = []
    for number in range(1, runs + 1):
        started = time.perf_counter()
        try:
            text, inference_ms = recognizer.recognize(image)
            if not isinstance(text, str):
                reads.append(
                    {
                        "run": number,
                        "status": "invalid_response",
                        "text": None,
                        "raw_response": None,
                        "error": "recognizer returned non-text output",
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
            else:
                reads.append(
                    {
                        "run": number,
                        "status": "ok",
                        "text": text,
                        "raw_response": None,
                        "inference_ms": round(inference_ms, 3),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
        except (OSError, RuntimeError, ValueError) as error:
            reads.append(
                {
                    "run": number,
                    "status": "request_failure",
                    "text": None,
                    "raw_response": None,
                    "error": str(error),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    distinct = list(dict.fromkeys(item["text"] for item in reads if item["status"] == "ok"))
    return {
        "model": metadata.get("model", TROCR_MODEL_ID),
        "configuration": metadata,
        "runs": reads,
        "parsed_readings": [item["text"] for item in reads if isinstance(item.get("text"), str)],
        "normalized_readings": [
            _normalize(item["text"]) for item in reads if isinstance(item.get("text"), str)
        ],
        "distinct_readings": distinct,
        "stable": len(distinct) == 1 and bool(distinct),
        "failures": [item for item in reads if item["status"] != "ok"],
    }


def _candidate_summary(results: list[dict[str, Any]], recognizer: str) -> dict[str, Any]:
    evidence: dict[str, dict[str, Any]] = {}
    for result in results:
        for reading in result[recognizer]["runs"]:
            if reading["status"] != "ok" or not isinstance(reading.get("text"), str):
                continue
            candidate = reading["text"]
            entry = evidence.setdefault(candidate, {"variants": [], "runs": 0})
            if result["variant"] not in entry["variants"]:
                entry["variants"].append(result["variant"])
            entry["runs"] += 1
    ordered = sorted(
        evidence.items(), key=lambda item: (-item[1]["runs"], item[1]["variants"][0], item[0])
    )
    return {
        "recognizer": recognizer,
        "candidate": ordered[0][0] if ordered else None,
        "candidates": [
            {"candidate": candidate, "support": support} for candidate, support in ordered
        ],
    }


def _compare_variants(results: list[dict[str, Any]], recognizer: str) -> dict[str, Any]:
    normalized_sets = [set(result[recognizer]["normalized_readings"]) for result in results]
    stable = (
        sorted(set.intersection(*normalized_sets))
        if normalized_sets and all(normalized_sets)
        else []
    )
    appearances: dict[str, list[str]] = {}
    for result in results:
        for reading in set(result[recognizer]["normalized_readings"]):
            appearances.setdefault(reading, []).append(result["variant"])
    return {
        "recognizer": recognizer,
        "stable_across_all_variants": stable,
        "variant_appearances": appearances,
        "variance_by_variant": {
            result["variant"]: len(result[recognizer]["distinct_readings"]) for result in results
        },
    }


def _load_review(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Review file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        ground_truth = value.get("ground_truth") if isinstance(value, dict) else None
        if isinstance(ground_truth, dict):
            return {"ground_truth": ground_truth}
    except json.JSONDecodeError:
        pass
    text = None
    confirmed = None
    in_ground_truth = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "ground_truth:":
            in_ground_truth = True
        elif in_ground_truth and stripped.startswith("text:"):
            text = stripped.split(":", 1)[1].strip().strip("\"'")
        elif in_ground_truth and stripped.startswith("confirmed:"):
            confirmed = stripped.split(":", 1)[1].strip().lower() == "true"
    if not isinstance(text, str) or not isinstance(confirmed, bool):
        raise ValueError("review requires ground_truth.text and ground_truth.confirmed")
    return {"ground_truth": {"text": text, "confirmed": confirmed}}


def run_handwriting_preprocess(
    image_path: Path,
    *,
    runs: int | None = None,
    regions: str | None = None,
    lines: str | None = None,
    crop_path: Path | None = None,
    review_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    compare_artifact: Path = DEFAULT_COMPARE_ARTIFACT,
    reread_artifact: Path = DEFAULT_REREAD_ARTIFACT,
    reader_factory: ReaderFactory | None = None,
    trocr_factory: TrocrFactory | None = None,
) -> dict[str, Any]:
    requested = requested_runs(runs)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    targets = _targets(
        image_path,
        regions=regions,
        lines=lines,
        crop_path=crop_path,
        compare_path=compare_artifact,
        reread_path=reread_artifact,
    )
    review = _load_review(review_path)
    try:
        with Image.open(image_path) as source_file:
            source_hash = _sha256(image_path)
        source_dimensions = {"width": source_file.width, "height": source_file.height}
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error
    trocr: TrocrReader | None = None
    trocr_metadata: dict[str, Any] = {}
    trocr_error: Exception | None = None
    try:
        if trocr_factory:
            trocr, trocr_metadata = trocr_factory()
        else:
            trocr, load_ms, cuda, device, gpu = TrocrRecognizer.from_local_cache()
            trocr_metadata = {
                "model": TROCR_MODEL_ID,
                "load_ms": round(load_ms, 3),
                "cuda_available": cuda,
                "device": device,
                "gpu": gpu,
                "processor": "TrOCRProcessor",
                "internal_resize": True,
            }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        trocr_error = error
    artifact: dict[str, Any] = {
        "experiment": "handwriting_preprocess",
        "source": str(image_path),
        "source_sha256": source_hash,
        "source_dimensions": source_dimensions,
        "runs_requested": requested,
        "regions_selected": sorted(selected_ids(regions) or []),
        "lines_selected": sorted(selected_ids(lines) or []),
        "review_input": str(review_path) if review_path else None,
        "review": review,
        "trocr": {"model": TROCR_MODEL_ID, "metadata": trocr_metadata},
        "regions": [],
        "output": {"json": str(output_path)},
    }
    try:
        for target in targets:
            crop = target["path"]
            source_crop_hash = _sha256(crop)
            with Image.open(crop) as source_crop_file:
                source_crop = source_crop_file.convert("RGB")
            variant_dir = output_path.parent / "handwriting-preprocess" / target["target_id"]
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_results: list[dict[str, Any]] = []
            for name, variant, parameters in _variant_images(source_crop):
                variant_path = variant_dir / f"{name}.png"
                if name == "rgb-original":
                    shutil.copyfile(crop, variant_path)
                else:
                    variant.save(variant_path, format="PNG")
                qwen = _run_qwen(variant, requested, reader_factory)
                if trocr_error is not None or trocr is None:
                    trocr_result = {
                        "status": "model_unavailable",
                        "model": TROCR_MODEL_ID,
                        "configuration": trocr_metadata,
                        "runs": [],
                        "parsed_readings": [],
                        "normalized_readings": [],
                        "distinct_readings": [],
                        "stable": False,
                        "failures": [
                            {
                                "status": "model_unavailable",
                                "error": str(trocr_error or "TrOCR unavailable"),
                            }
                        ],
                    }
                else:
                    trocr_result = _run_trocr(variant, requested, trocr, trocr_metadata)
                variant_results.append(
                    {
                        "variant": name,
                        "source_dimensions": {
                            "width": source_crop.width,
                            "height": source_crop.height,
                        },
                        "dimensions": {"width": variant.width, "height": variant.height},
                        "prepared_dimensions": {"width": variant.width, "height": variant.height},
                        "parameters": parameters,
                        "path": str(variant_path),
                        "sha256": _sha256(variant_path),
                        "qwen": qwen,
                        "trocr": trocr_result,
                    }
                )
                variant.close()
            region_result: dict[str, Any] = {
                "target_id": target["target_id"],
                "kind": target["kind"],
                "source_crop": {
                    "path": str(crop),
                    "sha256": source_crop_hash,
                    "artifact_sha256": target["metadata"].get("sha256"),
                    "identity_verified": target["metadata"].get("sha256")
                    in (None, source_crop_hash),
                    "dimensions": {"width": source_crop.width, "height": source_crop.height},
                    "mode": "RGB",
                    "original_pixels_available": True,
                    "source_bbox": target["source_bbox"],
                    "source_coordinate_space": target["source_coordinate_space"],
                    "source_artifact": target["source_artifact"],
                    "metadata": target["metadata"],
                },
                "variants": variant_results,
                "comparison": {
                    "qwen": _compare_variants(variant_results, "qwen"),
                    "trocr": _compare_variants(variant_results, "trocr"),
                    "cross_model_agreement": sorted(
                        set(
                            item
                            for result in variant_results
                            for item in result["qwen"]["normalized_readings"]
                            if item in result["trocr"]["normalized_readings"]
                        )
                    ),
                },
                "candidates": {
                    "qwen": _candidate_summary(variant_results, "qwen"),
                    "trocr": _candidate_summary(variant_results, "trocr"),
                },
            }
            if (
                review
                and isinstance(review.get("ground_truth"), dict)
                and review["ground_truth"].get("confirmed") is True
            ):
                truth = review["ground_truth"].get("text")
                if isinstance(truth, str):
                    region_result["evaluation"] = {
                        name: evaluate_candidate(
                            region_result["candidates"][name]["candidate"], truth
                        )
                        for name in ("qwen", "trocr")
                        if region_result["candidates"][name]["candidate"] is not None
                    }
            artifact["regions"].append(region_result)
            source_crop.close()
    finally:
        if trocr is not None and hasattr(trocr, "release"):
            trocr.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def format_handwriting_preprocess_result(artifact: dict[str, Any]) -> str:
    lines = [
        f"experiment: {artifact['experiment']}",
        f"source: {artifact['source']}",
        f"output: {artifact.get('output', artifact.get('artifact', ''))}",
        f"targets: {len(artifact['regions'])}",
    ]
    for region in artifact["regions"]:
        lines.extend(
            [
                f"{region['target_id']}:",
                f"  crop: {region['source_crop']['path']}",
                f"  variants: {len(region['variants'])}",
                f"  Qwen candidate: {region['candidates']['qwen']['candidate']}",
                f"  TrOCR candidate: {region['candidates']['trocr']['candidate']}",
            ]
        )
    return "\n".join(lines)

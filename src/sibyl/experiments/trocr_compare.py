"""Compare Qwen and TrOCR using one persisted source-resolution crop per region."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image, UnidentifiedImageError

from sibyl.experiments.transcription_reread import (
    REGIONAL_PROMPT,
    REGIONAL_SCHEMA,
    OllamaRegionalReader,
    OllamaTextRegionLocalizer,
    RegionalReader,
    TextRegionLocalizer,
    _reading_summary,
    _source_crop,
    deduplicate_regions,
    requested_runs,
    validate_regions,
)
from sibyl.experiments.trocr import MODEL_ID as TROCR_MODEL_ID
from sibyl.experiments.trocr import TrocrRecognizer
from sibyl.transform import DEFAULT_QWEN_MODEL, prepare_page_image_with_metadata

DEFAULT_OUTPUT = Path(".sibyl/experiments/trocr-compare.json")


class TrocrReader(Protocol):
    def recognize(self, image: Image.Image) -> tuple[str, float]: ...


def selected_regions(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise ValueError("--regions must contain at least one region ID")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_qwen(
    image: Image.Image,
    runs: int,
    reader_factory: Callable[[Callable[[dict[str, Any]], None]], RegionalReader] | None,
) -> tuple[dict[str, Any], str]:
    raw: Any = None

    def observe(value: dict[str, Any]) -> None:
        nonlocal raw
        raw = value

    reader = reader_factory(observe) if reader_factory else OllamaRegionalReader(observer=observe)
    results: list[dict[str, Any]] = []
    try:
        for number in range(1, runs + 1):
            raw = None
            started = time.perf_counter()
            try:
                value, duration_ms = reader.read(image)
                response = raw if raw is not None else value.get("raw_response")
                if value.get("status") == "invalid_response" or not isinstance(
                    value.get("text"), str
                ):
                    results.append(
                        {
                            "run": number,
                            "status": "invalid_response",
                            "text": None,
                            "raw_response": response,
                            "error": value.get("error", "missing text"),
                            "duration_ms": round(duration_ms, 3),
                        }
                    )
                else:
                    results.append(
                        {
                            "run": number,
                            "status": "ok",
                            "text": value["text"],
                            "raw_response": response,
                            "duration_ms": round(duration_ms, 3),
                        }
                    )
            except (RuntimeError, ValueError) as error:
                results.append(
                    {
                        "run": number,
                        "status": "request_failure",
                        "text": None,
                        "raw_response": raw,
                        "error": str(error),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
    finally:
        reader.release()
    summary = _reading_summary(results)
    return {
        "model": getattr(reader, "model", DEFAULT_QWEN_MODEL),
        "request_controls": {
            "num_predict": 256,
            "think": False,
            "stream": False,
            "keep_alive": 0,
            "prompt": REGIONAL_PROMPT,
            "schema": REGIONAL_SCHEMA,
        },
        "runs": results,
        "distinct_readings": summary["distinct_readings"],
        "stable": summary["stable"],
    }, getattr(reader, "model", DEFAULT_QWEN_MODEL)


def _run_trocr(
    image: Image.Image,
    runs: int,
    recognizer: TrocrReader,
    *,
    model: str,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for number in range(1, runs + 1):
        started = time.perf_counter()
        try:
            text, inference_ms = recognizer.recognize(image)
            results.append(
                {
                    "run": number,
                    "status": "ok",
                    "text": text,
                    "inference_ms": round(inference_ms, 3),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except RuntimeError as error:
            results.append(
                {
                    "run": number,
                    "status": "request_failure",
                    "text": None,
                    "error": str(error),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except (OSError, ValueError) as error:
            results.append(
                {
                    "run": number,
                    "status": "invalid_response",
                    "text": None,
                    "error": str(error),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    summary = _reading_summary(results)
    return {
        "status": "ok",
        "model": model,
        "preprocessing": preprocessing,
        "runs": results,
        "distinct_readings": summary["distinct_readings"],
        "stable": summary["stable"],
    }


def _unavailable(error: Exception, model: str = TROCR_MODEL_ID) -> dict[str, Any]:
    return {
        "status": "model_unavailable",
        "model": model,
        "preprocessing": {"processor": "TrOCRProcessor", "internal_resize": True},
        "runs": [],
        "distinct_readings": [],
        "stable": False,
        "error": str(error),
    }


def _overlap(qwen: dict[str, Any], trocr: dict[str, Any]) -> list[str]:
    return [item for item in qwen["distinct_readings"] if item in trocr["distinct_readings"]]


def run_compare_experiment(
    image_path: Path,
    *,
    runs: int | None = None,
    regions: str | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    localizer_factory: Callable[[Callable[[dict[str, Any]], None]], TextRegionLocalizer]
    | None = None,
    reader_factory: Callable[[Callable[[dict[str, Any]], None]], RegionalReader] | None = None,
    trocr_factory: Callable[[], tuple[TrocrReader, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    requested = requested_runs(runs)
    wanted = selected_regions(regions)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as source_file:
            source = source_file.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error

    prepared = prepare_page_image_with_metadata(source)
    raw_localization: Any = None

    def observe(value: dict[str, Any]) -> None:
        nonlocal raw_localization
        raw_localization = value

    localizer = (
        localizer_factory(observe)
        if localizer_factory
        else OllamaTextRegionLocalizer(observer=observe)
    )
    try:
        try:
            localization, localization_ms = localizer.localize(prepared.image)
        except (RuntimeError, ValueError) as error:
            localization, localization_ms = {"status": "request_failure", "error": str(error)}, 0.0
    finally:
        localizer.release()
    accepted, rejected = validate_regions(localization.get("text_regions"))
    accepted, duplicates = deduplicate_regions(accepted)
    rejected.extend(duplicates)
    accepted.sort(key=lambda item: item["index"])
    if wanted is not None:
        available = {f"region-{located['index'] + 1:02d}" for located in accepted}
        missing = wanted - available
        if missing:
            raise ValueError(f"unknown region IDs: {', '.join(sorted(missing))}")

    trocr: TrocrReader | None = None
    trocr_metadata: dict[str, Any] = {}
    trocr_error: Exception | None = None
    if trocr_factory is not None:
        try:
            trocr, trocr_metadata = trocr_factory()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            trocr_error = error
    else:
        try:
            trocr, load_ms, cuda, device, gpu = TrocrRecognizer.from_local_cache()
            trocr_metadata = {
                "model": TROCR_MODEL_ID,
                "load_ms": round(load_ms, 3),
                "cuda_available": cuda,
                "device": device,
                "gpu": gpu,
                "processor": "TrOCRProcessor",
                "internal_resize": True,
                "decoding": "VisionEncoderDecoderModel.generate default decoding",
            }
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            trocr_error = error

    artifact: dict[str, Any] = {
        "experiment": "trocr_compare",
        "source": str(image_path),
        "runs_requested": requested,
        "regions_selected": sorted(wanted) if wanted else None,
        "localization": {
            "status": localization.get(
                "status", "ok" if "text_regions" in localization else "invalid_response"
            ),
            "error": localization.get("error"),
            "raw_response": raw_localization or localization.get("raw_response"),
            "duration_ms": round(localization_ms, 3),
            "rejected_regions": rejected,
        },
        "qwen": {"prompt_contract": "existing transcription-reread regional request"},
        "trocr": {"model": TROCR_MODEL_ID, "metadata": trocr_metadata},
        "regions": [],
    }
    for located in accepted:
        # Keep the localization identity stable across duplicate rejection.  A
        # later accepted region is still region-10, not region-06, when the
        # source localization contained rejected entries before it.
        region_id = f"region-{located['index'] + 1:02d}"
        if wanted is not None and region_id not in wanted:
            continue
        crop_info = _source_crop(source, prepared, located["bbox_2d"], output_path, region_id)
        crop_image = cast(Image.Image, crop_info.pop("image"))
        crop_path = Path(crop_info["path"])
        crop_info.update({"sha256": _sha256(crop_path), "mode": crop_image.mode})
        if crop_image.mode != "RGB":
            raise ValueError("source crop was not persisted as RGB")
        if trocr_error is not None or trocr is None:
            trocr_result = _unavailable(trocr_error or RuntimeError("TrOCR unavailable"))
        else:
            trocr_result = _run_trocr(
                crop_image,
                requested,
                trocr,
                model=trocr_metadata.get("model", TROCR_MODEL_ID),
                preprocessing=trocr_metadata,
            )
        qwen_result, _ = _run_qwen(crop_image, requested, reader_factory)
        artifact["regions"].append(
            {
                "region_id": region_id,
                "crop": crop_info,
                "qwen_input_hash": crop_info["sha256"],
                "trocr_input_hash": crop_info["sha256"],
                "qwen": qwen_result,
                "trocr": trocr_result,
                "comparison": {
                    "qwen_distinct_readings": qwen_result["distinct_readings"],
                    "trocr_distinct_readings": trocr_result["distinct_readings"],
                    "overlap": _overlap(qwen_result, trocr_result),
                    "qwen_stable": qwen_result["stable"],
                    "trocr_stable": trocr_result["stable"],
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def format_compare_result(artifact: dict[str, Any]) -> str:
    lines = [
        f"experiment: {artifact['experiment']}",
        f"source: {artifact['source']}",
        "",
        f"regions: {len(artifact['regions'])}",
    ]
    for region in artifact["regions"]:
        lines.extend(
            [
                "",
                f"{region['region_id']}:",
                f"  crop: {region['crop']['path']}",
                f"  crop hash: {region['crop']['sha256']}",
                "  Qwen:",
            ]
        )
        lines.extend(
            f"    {item['run']}: {item['text'] or '[' + item['status'] + ']'}"
            for item in region["qwen"]["runs"]
        )
        lines.append(f"    distinct: {len(region['qwen']['distinct_readings'])}")
        lines.append("  TrOCR:")
        lines.extend(
            f"    {item['run']}: {item['text'] or '[' + item['status'] + ']'}"
            for item in region["trocr"]["runs"]
        )
        lines.append(f"    distinct: {len(region['trocr']['distinct_readings'])}")
        lines.append(f"  cross-model overlap: {region['comparison']['overlap'] or 'none'}")
    return "\n".join(lines)

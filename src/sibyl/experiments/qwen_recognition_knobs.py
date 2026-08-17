"""Controlled Qwen prompt, context, and decoding observations."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from sibyl.experiments.handwriting_preprocess import _targets, evaluate_candidate
from sibyl.experiments.transcription_reread import (
    REGIONAL_PROMPT,
    REGIONAL_SCHEMA,
    _message_json,
)
from sibyl.transform import DEFAULT_OLLAMA_URL, DEFAULT_QWEN_MODEL

DEFAULT_OUTPUT = Path(".sibyl/experiments/qwen-recognition-knobs.json")
DEFAULT_RUNS = 5
DEFAULT_NUM_PREDICT = 256
ISOLATED_PROMPT = (
    "Read the handwritten text in this image.\nReturn only the transcription.\n"
    "Do not describe the image.\nDo not infer missing words.\nPreserve the observed spelling."
)
EXACT_WORD_PROMPT = (
    "Read the handwritten word or words in this image.\nReturn only the handwritten text.\n"
    "Do not describe the image.\nDo not infer text that is not visible."
)
PROMPTS = {
    "regional": REGIONAL_PROMPT,
    "isolated": ISOLATED_PROMPT,
    "exact-word": EXACT_WORD_PROMPT,
}
DEFAULT_TEMPERATURES = (0.0, 0.2, 0.5, 0.8)
DEFAULT_TOP_P = (1.0, 0.9)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


class KnobReader(Protocol):
    model: str

    def read(
        self, image: Image.Image, prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


ReaderFactory = Callable[[Callable[[dict[str, Any]], None]], KnobReader]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_data(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _query(
    *,
    model: str,
    base_url: str,
    image: Image.Image,
    prompt: str,
    controls: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    options = {key: value for key, value in controls.items() if value is not None}
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "format": REGIONAL_SCHEMA,
        "options": options,
        "messages": [{"role": "user", "content": prompt, "images": [_image_data(image)]}],
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to query Ollama/Qwen ({model}): {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("Ollama/Qwen returned a non-object response")
    return body, (time.perf_counter() - started) * 1000


class OllamaKnobReader:
    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        import os

        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        self.base_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.observer = observer

    def read(
        self, image: Image.Image, prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        body, duration = _query(
            model=self.model, base_url=self.base_url, image=image, prompt=prompt, controls=controls
        )
        if self.observer is not None:
            self.observer(body)
        text = extract_recognition_text(body)
        truncated = body.get("done_reason") == "length"
        if text is None:
            return {
                "status": "truncated_response" if truncated else "invalid_response",
                "error": "response was truncated" if truncated else "missing text",
                "raw_response": body,
            }, duration
        if truncated:
            return {
                "status": "truncated_response",
                "text": text,
                "raw_response": body,
            }, duration
        return {"status": "ok", "text": text, "raw_response": body}, duration

    def release(self) -> None:
        return None


def extract_recognition_text(body: dict[str, Any]) -> str | None:
    """Extract recognition text in Sibyl's established content/thinking order."""
    parsed = _message_json(body)
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return cast(str, parsed["text"])
    message = body.get("message")
    if not isinstance(message, dict):
        return None
    for field in ("content", "thinking"):
        candidate = message.get(field)
        if isinstance(candidate, dict) and isinstance(candidate.get("text"), str):
            return cast(str, candidate["text"])
    return None


def _bounds(metadata: dict[str, Any], source: Image.Image) -> tuple[int, int, int, int] | None:
    value = metadata.get("source_bbox")
    if not isinstance(value, dict):
        return None
    try:
        return (int(value["left"]), int(value["top"]), int(value["right"]), int(value["bottom"]))
    except (KeyError, TypeError, ValueError):
        return None


def _context_variants(source: Image.Image, target: dict[str, Any]) -> list[dict[str, Any]]:
    bounds = _bounds(target, source)
    if bounds is None:
        with Image.open(target["path"]) as image:
            return [{"variant": "tight", "image": image.convert("RGB").copy(), "source_bbox": None}]
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    variants: list[dict[str, Any]] = []
    for label, proportion in (
        ("tight", 0.0),
        ("padding-05", 0.05),
        ("padding-10", 0.10),
        ("padding-20", 0.20),
        ("padding-30", 0.30),
    ):
        dx, dy = round(width * proportion), round(height * proportion)
        box = (
            max(0, left - dx),
            max(0, top - dy),
            min(source.width, right + dx),
            min(source.height, bottom + dy),
        )
        variants.append(
            {
                "variant": label,
                "image": source.crop(box).convert("RGB"),
                "source_bbox": dict(zip(("left", "top", "right", "bottom"), box, strict=True)),
            }
        )
    mapping = target.get("metadata", {}).get("mapping", {})
    coarse = mapping.get("coarse_source_bbox") if isinstance(mapping, dict) else None
    if isinstance(coarse, dict):
        coarse_box = (
            int(coarse["left"]),
            int(coarse["top"]),
            int(coarse["right"]),
            int(coarse["bottom"]),
        )
        variants.append(
            {
                "variant": "surrounding-region",
                "image": source.crop(coarse_box).convert("RGB"),
                "source_bbox": coarse,
            }
        )
    return variants


def _parse_values(values: str | None, default: Iterable[float]) -> tuple[float, ...]:
    if values is None:
        return tuple(default)
    try:
        result = tuple(float(item.strip()) for item in values.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("numeric sweep values must be comma-separated") from error
    if not result:
        raise ValueError("sweep must contain at least one value")
    return result


def _load_review(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        text: str | None = None
        confirmed: bool | None = None
        in_ground_truth = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped == "ground_truth:":
                in_ground_truth = True
            elif in_ground_truth and stripped.startswith("text:"):
                text = stripped.split(":", 1)[1].strip().strip("\"'")
            elif in_ground_truth and stripped.startswith("confirmed:"):
                confirmed = stripped.split(":", 1)[1].strip().lower() == "true"
        if not isinstance(text, str) or confirmed is None:
            raise ValueError(
                "review requires ground_truth.text and ground_truth.confirmed"
            ) from None
        return {"ground_truth": {"text": text, "confirmed": confirmed}}


def _readings(
    reader: KnobReader,
    image: Image.Image,
    prompt: str,
    controls: dict[str, Any],
    runs: int,
) -> dict[str, Any]:
    reads: list[dict[str, Any]] = []
    for number in range(1, runs + 1):
        started = time.perf_counter()
        try:
            value, duration = reader.read(image, prompt, controls)
            raw = value.get("raw_response")
            status = value.get("status", "ok")
            reads.append(
                {
                    "run": number,
                    "seed": controls.get("seed"),
                    "status": status,
                    "reading": value.get("text") if status == "ok" else None,
                    "raw_response": raw,
                    "error": value.get("error"),
                    "duration_ms": round(duration, 3),
                    "token_counts": raw.get("prompt_eval_count") if isinstance(raw, dict) else None,
                }
            )
        except (RuntimeError, ValueError, OSError) as error:
            reads.append(
                {
                    "run": number,
                    "status": "provider_failure",
                    "reading": None,
                    "raw_response": None,
                    "error": str(error),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "token_counts": None,
                }
            )
    observed = [item["reading"] for item in reads if isinstance(item.get("reading"), str)]
    normalized = [_normalize(item) for item in observed]
    counts = Counter(normalized)
    first = {candidate: normalized.index(candidate) + 1 for candidate in counts}
    per_seed: dict[str, list[str]] = {}
    for item in reads:
        if item.get("status") == "ok" and isinstance(item.get("reading"), str):
            per_seed.setdefault(str(item.get("seed")), []).append(_normalize(item["reading"]))
    return {
        "runs": reads,
        "readings": observed,
        "normalized_readings": normalized,
        "distinct_readings": list(dict.fromkeys(observed)),
        "candidates": [
            {
                "candidate": candidate,
                "normalized": candidate,
                "frequency": counts[candidate],
                "first_occurrence": first[candidate],
                "stability": counts[candidate] / runs,
            }
            for candidate in sorted(counts)
        ],
        "per_seed": per_seed,
        "failures": [item for item in reads if item.get("status") != "ok"],
    }


def aggregate_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for result in results:
        for candidate in result.get("analysis", {}).get("candidates", []):
            normalized = candidate["normalized"]
            entry = aggregate.setdefault(
                normalized, {"candidate": candidate["candidate"], "frequency": 0, "support": []}
            )
            entry["frequency"] += candidate["frequency"]
            entry["support"].append(
                {
                    "prompt_variant": result["prompt_variant"],
                    "context_variant": result["context_variant"],
                    "decoding_controls": result["decoding_controls"],
                }
            )
    return sorted(aggregate.values(), key=lambda item: (-item["frequency"], item["candidate"]))


def run_qwen_recognition_knobs(
    image_path: Path,
    *,
    runs: int = DEFAULT_RUNS,
    regions: str | None = None,
    lines: str | None = None,
    crop_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    compare_artifact: Path = Path(".sibyl/experiments/trocr-compare.json"),
    reread_artifact: Path = Path(".sibyl/experiments/transcription-reread.json"),
    review_path: Path | None = None,
    prompt_variants: tuple[str, ...] = ("regional", "isolated", "exact-word"),
    contexts: tuple[str, ...] | None = None,
    temperatures: tuple[float, ...] = (0.0,),
    top_ps: tuple[float, ...] = (1.0,),
    seeds: tuple[int | None, ...] = (None,),
    num_predict: int = DEFAULT_NUM_PREDICT,
    reader_factory: ReaderFactory | None = None,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
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
    with Image.open(image_path) as source_file:
        source = source_file.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review: dict[str, Any] | None = None
    if review_path is not None:
        review = _load_review(review_path)
    control_matrix: list[dict[str, Any]] = [
        {"temperature": None, "top_p": None, "seed": None, "num_predict": num_predict}
    ]
    control_matrix.extend(
        {"temperature": temperature, "top_p": top_p, "seed": seed, "num_predict": num_predict}
        for temperature in temperatures
        for top_p in top_ps
        for seed in seeds
        if temperature != 0.0 or top_p != 1.0 or seed is not None
    )
    results: list[dict[str, Any]] = []
    artifact: dict[str, Any] = {
        "experiment": "qwen_recognition_knobs",
        "source": str(image_path),
        "source_sha256": _sha256(image_path),
        "source_dimensions": {"width": source.width, "height": source.height},
        "runs_requested": runs,
        "targets": [],
        "prompts": PROMPTS,
        "decoding_sweeps": {
            "temperatures": list(temperatures),
            "top_p": list(top_ps),
            "seeds": list(seeds),
        },
        "review": review,
        "results": [],
        "candidate_aggregation": [],
        "output": str(output_path),
    }
    for target in targets:
        contexts_for_target = _context_variants(source, target)
        if contexts is not None:
            contexts_for_target = [
                item for item in contexts_for_target if item["variant"] in contexts
            ]
        target_record = {
            "target_id": target["target_id"],
            "kind": target["kind"],
            "source_crop": {
                "path": str(target["path"]),
                "sha256": _sha256(target["path"]),
                "source_bbox": target["source_bbox"],
                "source_coordinate_space": target["source_coordinate_space"],
                "dimensions": {
                    "width": Image.open(target["path"]).width,
                    "height": Image.open(target["path"]).height,
                },
            },
            "contexts": [],
        }
        artifact["targets"].append(target_record)
        for context in contexts_for_target:
            context_path = (
                output_path.parent
                / "qwen-recognition-knobs"
                / target["target_id"]
                / f"{context['variant']}.png"
            )
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context["image"].save(context_path, format="PNG")
            context_record = {
                "variant": context["variant"],
                "path": str(context_path),
                "sha256": _sha256(context_path),
                "source_bbox": context["source_bbox"],
                "dimensions": {"width": context["image"].width, "height": context["image"].height},
            }
            target_record["contexts"].append(context_record)
            for prompt_variant in prompt_variants:
                if prompt_variant not in PROMPTS:
                    raise ValueError(f"unknown prompt variant: {prompt_variant}")
                for controls in control_matrix:
                    reader_raw: Any = None

                    def observe(value: dict[str, Any]) -> None:
                        nonlocal reader_raw
                        reader_raw = value

                    reader = (
                        reader_factory(observe)
                        if reader_factory
                        else OllamaKnobReader(observer=observe)
                    )
                    try:
                        analysis = _readings(
                            reader, context["image"], PROMPTS[prompt_variant], controls, runs
                        )
                        model = reader.model
                    finally:
                        reader.release()
                    result = {
                        "target_id": target["target_id"],
                        "prompt_variant": prompt_variant,
                        "prompt": PROMPTS[prompt_variant],
                        "context_variant": context["variant"],
                        "crop_identity": {
                            "sha256": context_record["sha256"],
                            "dimensions": context_record["dimensions"],
                            "source_bbox": context["source_bbox"],
                        },
                        "model": model,
                        "decoding_controls": {
                            **controls,
                            "think": False,
                            "stream": False,
                            "keep_alive": 0,
                        },
                        "analysis": analysis,
                    }
                    results.append(result)
                    artifact["results"].append(result)
            context["image"].close()
    artifact["candidate_aggregation"] = aggregate_candidates(results)
    if (
        review
        and isinstance(review.get("ground_truth"), dict)
        and review["ground_truth"].get("confirmed") is True
    ):
        truth = review["ground_truth"].get("text")
        if isinstance(truth, str):
            artifact["evaluation"] = {
                "ground_truth": review["ground_truth"],
                "results": [
                    {
                        "target_id": result["target_id"],
                        "prompt_variant": result["prompt_variant"],
                        "context_variant": result["context_variant"],
                        "metrics": [
                            evaluate_candidate(reading, truth)
                            for reading in result["analysis"]["readings"]
                        ],
                    }
                    for result in results
                ],
            }
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def format_qwen_recognition_knobs(artifact: dict[str, Any]) -> str:
    return (
        f"experiment: {artifact['experiment']}\n"
        f"source: {artifact['source']}\n"
        f"results: {len(artifact['results'])}\n"
        f"output: {artifact.get('output', DEFAULT_OUTPUT)}"
    )

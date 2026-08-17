"""Inference-time visual-exemplar handwriting recognition experiment."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from sibyl.experiments.handwriting_preprocess import _targets, evaluate_candidate
from sibyl.experiments.qwen_recognition_knobs import (
    DEFAULT_NUM_PREDICT,
    ISOLATED_PROMPT,
    extract_recognition_text,
)
from sibyl.transform import DEFAULT_OLLAMA_URL, DEFAULT_QWEN_MODEL

DEFAULT_OUTPUT = Path(".sibyl/experiments/handwriting-exemplars.json")
DEFAULT_RUNS = 5
EXEMPLAR_PROMPT = (
    "REFERENCE IMAGES appear first, in the listed reference order. TARGET IMAGE appears last.\n"
    "Use the reference images only to understand the writer's handwriting style and glyph forms.\n"
    "Transcribe only the TARGET IMAGE.\n"
    "Do not copy words from the references unless the target visibly contains them.\n"
    "Return only the target transcription.\n"
    "Do not describe the images.\n"
    "Do not infer text that is not visible."
)


class ExemplarReader(Protocol):
    model: str

    def read(
        self, images: list[Image.Image], prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


ReaderFactory = Callable[[Callable[[dict[str, Any]], None]], ExemplarReader]


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
    images: list[Image.Image],
    prompt: str,
    controls: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {
            key: value
            for key, value in {
                "num_predict": controls["num_predict"],
                "num_ctx": controls.get("num_ctx"),
            }.items()
            if value is not None
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [_image_data(image) for image in images],
            }
        ],
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
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        suffix = f": {details}" if details else ""
        raise RuntimeError(
            f"Unable to query Ollama/Qwen ({model}); HTTP {error.code}{suffix}"
        ) from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to query Ollama/Qwen ({model}): {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("Ollama/Qwen returned a non-object response")
    return body, (time.perf_counter() - started) * 1000


class OllamaExemplarReader:
    def __init__(
        self,
        observer: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("SIBYL_QWEN_MODEL", DEFAULT_QWEN_MODEL)
        self.base_url = base_url or os.environ.get("SIBYL_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.observer = observer

    def read(
        self, images: list[Image.Image], prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        body, duration = _query(
            model=self.model,
            base_url=self.base_url,
            images=images,
            prompt=prompt,
            controls=controls,
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
        return {
            "status": "truncated_response" if truncated else "ok",
            "text": text,
            "raw_response": body,
        }, duration

    def release(self) -> None:
        return None


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.startswith("{") and value.endswith("}"):
        pairs: dict[str, Any] = {}
        for item in value[1:-1].split(","):
            key, separator, scalar = item.partition(":")
            if not separator:
                raise ValueError("invalid inline mapping in reference manifest")
            pairs[key.strip().strip("\"'")] = _parse_scalar(scalar)
        return pairs
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference manifest not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        references: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("- id:"):
                current = {"id": _parse_scalar(stripped.split(":", 1)[1])}
                references.append(current)
            elif current is not None and ":" in stripped:
                key, scalar = stripped.split(":", 1)
                current[key.strip()] = _parse_scalar(scalar)
        value = {"references": references}
    if not isinstance(value, dict) or not isinstance(value.get("references"), list):
        raise ValueError("reference manifest requires a references list")
    references = [item for item in value["references"] if isinstance(item, dict)]
    if len(references) != len(value["references"]):
        raise ValueError("every reference manifest entry must be an object")
    return references


def _reference_records(
    manifest_path: Path,
    selected: str | None,
    target_path: Path,
    target_hash: str,
) -> list[dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    for item in manifest:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("each reference requires a non-empty id")
        if identifier in by_id:
            raise ValueError(f"duplicate reference id: {identifier}")
        crop_value = item.get("crop")
        transcription = item.get("transcription")
        if not isinstance(crop_value, str) or not isinstance(transcription, str):
            raise ValueError(f"reference {identifier} requires crop and transcription")
        if item.get("confirmed") is not True:
            raise ValueError(f"reference {identifier} must be human-confirmed")
        source_bbox = item.get("source_bbox")
        if not isinstance(source_bbox, dict):
            raise ValueError(f"reference {identifier} requires source_bbox provenance")
        crop_path = Path(crop_value)
        if not crop_path.is_absolute():
            crop_path = manifest_path.parent / crop_path
        if not crop_path.is_file():
            raise FileNotFoundError(f"Reference crop not found: {crop_path}")
        crop_hash = _sha256(crop_path)
        if crop_path.resolve() == target_path.resolve() or crop_hash == target_hash:
            raise ValueError(f"reference {identifier} is the target crop")
        with Image.open(crop_path) as image:
            dimensions = {"width": image.width, "height": image.height}
        by_id[identifier] = {
            "reference_id": identifier,
            "path": crop_path,
            "sha256": crop_hash,
            "source_bbox": source_bbox,
            "source_coordinate_space": item.get("source_coordinate_space", "source"),
            "dimensions": dimensions,
            "transcription": transcription,
            "confirmed": True,
        }
    selected_ids = (
        sorted({item.strip() for item in selected.split(",") if item.strip()})
        if selected is not None
        else sorted(by_id)
    )
    if selected is not None and len(selected_ids) != len(
        [item for item in selected.split(",") if item.strip()]
    ):
        raise ValueError("duplicate reference selection")
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise ValueError(f"unknown reference IDs: {', '.join(sorted(missing))}")
    return [by_id[identifier] for identifier in selected_ids]


def _reference_sets(
    references: list[dict[str, Any]], explicit: str | None
) -> list[list[dict[str, Any]]]:
    baseline: list[list[dict[str, Any]]] = [[]]
    if explicit is not None:
        ids = [item.strip() for item in explicit.split(",") if item.strip()]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate reference-set selection")
        by_id = {item["reference_id"]: item for item in references}
        if set(ids) != set(by_id):
            raise ValueError("reference-set IDs must be selected by --references")
        return [*baseline, [by_id[identifier] for identifier in sorted(ids)]]
    return [*baseline, *[references[:size] for size in (1, 3, 5) if len(references) >= size]]


def _read_configuration(
    reader: ExemplarReader,
    images: list[Image.Image],
    prompt: str,
    controls: dict[str, Any],
    runs: int,
) -> dict[str, Any]:
    reads: list[dict[str, Any]] = []
    for number in range(1, runs + 1):
        started = time.perf_counter()
        try:
            result, duration = reader.read(images, prompt, controls)
            raw = result.get("raw_response")
            status = result.get("status", "ok")
            reads.append(
                {
                    "run": number,
                    "status": status,
                    "reading": result.get("text") if status == "ok" else None,
                    "raw_response": raw,
                    "error": result.get("error"),
                    "duration_ms": round(duration, 3),
                    "token_counts": {
                        "prompt": raw.get("prompt_eval_count"),
                        "output": raw.get("eval_count"),
                    }
                    if isinstance(raw, dict)
                    else None,
                }
            )
        except (OSError, RuntimeError, ValueError) as error:
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
    normalized = [" ".join(item.strip().casefold().split()) for item in observed]
    counts = Counter(normalized)
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
                "stability": counts[candidate] / runs,
            }
            for candidate in sorted(counts)
        ],
        "invalid_count": sum(item["status"] == "invalid_response" for item in reads),
        "truncated_count": sum(item["status"] == "truncated_response" for item in reads),
        "failures": [item for item in reads if item["status"] != "ok"],
    }


def _copy_image(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)


def run_handwriting_exemplars(
    image_path: Path,
    *,
    regions: str | None = None,
    lines: str | None = None,
    target_crop: Path | None = None,
    references: str | None = None,
    reference_manifest: Path | None = None,
    reference_set: str | None = None,
    runs: int = DEFAULT_RUNS,
    review_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    compare_artifact: Path = Path(".sibyl/experiments/trocr-compare.json"),
    reread_artifact: Path = Path(".sibyl/experiments/transcription-reread.json"),
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
        crop_path=target_crop,
        compare_path=compare_artifact,
        reread_path=reread_artifact,
    )
    if len(targets) != 1:
        raise ValueError("handwriting-exemplars requires exactly one target")
    target = targets[0]
    target_path = Path(target["path"])
    target_hash = _sha256(target_path)
    selected_references: list[dict[str, Any]] = []
    if references is not None or reference_set is not None:
        if reference_manifest is None:
            raise ValueError("--references requires --reference-manifest")
        selected_references = _reference_records(
            reference_manifest, references, target_path, target_hash
        )
    if reference_set is not None and not selected_references:
        raise ValueError("--reference-set requires selected references")
    reference_sets = _reference_sets(selected_references, reference_set)
    review: dict[str, Any] | None = None
    if review_path is not None:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    controls = {
        "model": DEFAULT_QWEN_MODEL,
        "temperature": "unspecified (Ollama/model default)",
        "top_p": "unspecified (Ollama/model default)",
        "seed": "unspecified (Ollama/model default)",
        "num_predict": DEFAULT_NUM_PREDICT,
        "think": False,
        "stream": False,
        "keep_alive": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source_file:
        source_dimensions = {"width": source_file.width, "height": source_file.height}
    generated_dir = output_path.parent / "handwriting-exemplars" / target["target_id"]
    generated_target = generated_dir / "target.png"
    _copy_image(target_path, generated_target)
    generated_references: dict[str, Path] = {}
    for reference in selected_references:
        generated_path = generated_dir / f"{reference['reference_id']}.png"
        _copy_image(reference["path"], generated_path)
        generated_references[reference["reference_id"]] = generated_path
    artifact: dict[str, Any] = {
        "experiment": "handwriting_exemplars",
        "source": str(image_path),
        "source_sha256": _sha256(image_path),
        "source_dimensions": source_dimensions,
        "target": {
            "target_id": target["target_id"],
            "kind": target["kind"],
            "path": str(target_path),
            "sha256": target_hash,
            "source_bbox": target["source_bbox"],
            "source_coordinate_space": target["source_coordinate_space"],
            "generated_path": str(generated_target),
            "generated_sha256": _sha256(generated_target),
        },
        "references": [
            {
                **{key: value for key, value in reference.items() if key != "path"},
                "generated_path": str(generated_references[reference["reference_id"]]),
                "generated_sha256": _sha256(generated_references[reference["reference_id"]]),
            }
            | {"path": str(reference["path"])}
            for reference in selected_references
        ],
        "reference_sets": [],
        "runs_requested": runs,
        "request_controls": controls,
        "review": review,
        "results": [],
        "output": str(output_path),
    }
    target_image = Image.open(target_path).convert("RGB")
    try:
        for reference_set_items in reference_sets:
            set_ids = [item["reference_id"] for item in reference_set_items]
            set_name = "baseline" if not set_ids else f"references-{len(set_ids):02d}"
            set_record = {"set_id": set_name, "reference_ids": set_ids}
            artifact["reference_sets"].append(set_record)
            exemplar_images = [
                Image.open(item["path"]).convert("RGB") for item in reference_set_items
            ]
            images = [*exemplar_images, target_image]
            prompt = ISOLATED_PROMPT if not set_ids else EXEMPLAR_PROMPT
            reader_raw: Any = None

            def observe(value: dict[str, Any]) -> None:
                nonlocal reader_raw
                reader_raw = value

            reader = (
                reader_factory(observe)
                if reader_factory
                else OllamaExemplarReader(observer=observe)
            )
            try:
                analysis = _read_configuration(reader, images, prompt, controls, runs)
                model = reader.model
            finally:
                reader.release()
            artifact["results"].append(
                {
                    "reference_set": set_record,
                    "prompt_variant": "baseline" if not set_ids else "visual-exemplars",
                    "prompt": prompt,
                    "model": model,
                    "target_identity": artifact["target"],
                    "reference_identities": [
                        {
                            **{key: value for key, value in item.items() if key != "path"},
                            "generated_path": str(generated_references[item["reference_id"]]),
                            "generated_sha256": _sha256(generated_references[item["reference_id"]]),
                        }
                        | {"path": str(item["path"])}
                        for item in reference_set_items
                    ],
                    "image_order": [*set_ids, target["target_id"]],
                    "request_controls": controls,
                    "analysis": analysis,
                }
            )
            artifact["reference_sets"][-1]["raw_response_observed"] = reader_raw is not None
            for image in exemplar_images:
                image.close()
    finally:
        target_image.close()
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
                        "reference_set": result["reference_set"],
                        "metrics": [
                            evaluate_candidate(reading, truth)
                            for reading in result["analysis"]["readings"]
                        ],
                    }
                    for result in artifact["results"]
                ],
            }
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def format_handwriting_exemplars(artifact: dict[str, Any]) -> str:
    return (
        f"experiment: {artifact['experiment']}\n"
        f"target: {artifact['target']['target_id']}\n"
        f"reference_sets: {len(artifact['reference_sets'])}\n"
        f"output: {artifact['output']}"
    )

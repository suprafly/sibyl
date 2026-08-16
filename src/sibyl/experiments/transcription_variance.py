"""Measure repeated page-transcription variance without running the transform pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from sibyl.transform import (
    DEFAULT_QWEN_MODEL,
    OllamaPageInterpreter,
    PreparedVlmImage,
    prepare_page_image_with_metadata,
)

DEFAULT_RUNS = 5
DEFAULT_OUTPUT = Path(".sibyl/experiments/transcription-variance.json")


class PageTranscriber(Protocol):
    model: str
    response_metadata: dict[str, Any]

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class VarianceRun:
    run: int
    status: str
    text: str | None
    raw_response: Any
    duration_ms: float
    lines: list[str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class VarianceResult:
    experiment: str
    runs_requested: int
    runs_completed: int
    source: str
    page_focus: str
    model: str
    prepared_dimensions: dict[str, int]
    prepared_image_hash: str
    request_controls: dict[str, Any]
    runs: list[VarianceRun]
    comparison: dict[str, Any]
    failure_summary: dict[str, int]


PreparedObserver = Callable[[Image.Image, PreparedVlmImage], None]


def requested_runs(value: int | None = None) -> int:
    """Resolve and validate the requested run count without doing any inference."""
    configured = value
    if configured is None:
        raw = os.environ.get("SIBYL_TRANSCRIPTION_RUNS", str(DEFAULT_RUNS))
        try:
            configured = int(raw)
        except ValueError as error:
            raise ValueError("SIBYL_TRANSCRIPTION_RUNS must be a positive integer") from error
    if configured <= 0:
        raise ValueError("transcription variance runs must be a positive integer")
    return configured


def _prepared_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def prepared_image_hash(prepared: PreparedVlmImage) -> str:
    """Hash the exact lossless PNG representation handed to every request."""
    return hashlib.sha256(_prepared_png(prepared.image)).hexdigest()


def _page_text(interpretation: dict[str, Any]) -> str:
    page_text = interpretation.get("page_text")
    if not isinstance(page_text, list):
        page_interpretation = interpretation.get("page_interpretation", {})
        page_text = (
            page_interpretation.get("text", []) if isinstance(page_interpretation, dict) else []
        )
    if not isinstance(page_text, list) or not all(isinstance(item, str) for item in page_text):
        raise ValueError("valid structured response did not contain page text")
    return "\n".join(page_text)


def _page_lines(interpretation: dict[str, Any]) -> list[str]:
    page_text = interpretation.get("page_text")
    if not isinstance(page_text, list):
        page_interpretation = interpretation.get("page_interpretation", {})
        page_text = (
            page_interpretation.get("text", []) if isinstance(page_interpretation, dict) else []
        )
    if not isinstance(page_text, list) or not all(isinstance(item, str) for item in page_text):
        raise ValueError("valid structured response did not contain page text")
    return page_text


def _raw_from_interpreter(interpreter: PageTranscriber, interpretation: dict[str, Any]) -> Any:
    if "raw_response" in interpretation:
        return interpretation["raw_response"]
    return interpreter.response_metadata.get("raw_response")


def _comparison(runs: list[VarianceRun]) -> dict[str, Any]:
    successful = [run.text for run in runs if run.status == "ok" and run.text is not None]
    unique = list(dict.fromkeys(successful))
    return {
        "successful_runs": len(successful),
        "transcriptions_identical": len(unique) <= 1 and bool(successful),
        "transcriptions_different": len(unique) > 1,
        "outputs": unique,
    }


def run_variance_experiment(
    image_path: Path,
    *,
    runs: int | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    interpreter_factory: Callable[[Callable[[dict[str, Any]], None]], PageTranscriber]
    | None = None,
    prepared_observer: PreparedObserver | None = None,
    write_output: bool = True,
) -> VarianceResult:
    """Run repeated page transcription on one prepared image, never drawing localization."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    requested = requested_runs(runs)
    try:
        with Image.open(image_path) as source_file:
            source = source_file.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error

    prepared = prepare_page_image_with_metadata(source)
    if prepared_observer is not None:
        prepared_observer(source, prepared)
    image_hash = prepared_image_hash(prepared)
    raw_response: Any = None

    def observe(response: dict[str, Any]) -> None:
        nonlocal raw_response
        raw_response = response

    interpreter = (
        interpreter_factory(observe)
        if interpreter_factory is not None
        else OllamaPageInterpreter(response_observer=observe)
    )
    run_results: list[VarianceRun] = []
    try:
        for number in range(1, requested + 1):
            raw_response = None
            started = time.perf_counter()
            try:
                interpretation, duration_ms = interpreter.interpret(prepared.image)
                raw = raw_response
                if raw is None:
                    raw = _raw_from_interpreter(interpreter, interpretation)
                if interpretation.get("status") == "failure":
                    run_results.append(
                        VarianceRun(
                            run=number,
                            status="invalid_response",
                            text=None,
                            raw_response=raw,
                            duration_ms=round(duration_ms, 3),
                            error=str(interpretation.get("error", "invalid response")),
                        )
                    )
                else:
                    run_results.append(
                        VarianceRun(
                            run=number,
                            status="ok",
                            text=_page_text(interpretation),
                            raw_response=raw,
                            duration_ms=round(duration_ms, 3),
                            lines=_page_lines(interpretation),
                        )
                    )
            except (RuntimeError, ValueError) as error:
                run_results.append(
                    VarianceRun(
                        run=number,
                        status="failed",
                        text=None,
                        raw_response=raw_response,
                        duration_ms=round((time.perf_counter() - started) * 1000, 3),
                        error=str(error),
                    )
                )
    finally:
        interpreter.release()

    controls = {
        "model": interpreter.model,
        "think": False,
        "num_predict": 256,
        "stream": False,
        "keep_alive": 0,
        "temperature": "unspecified (Ollama/model default)",
        "top_p": "unspecified (Ollama/model default)",
        "seed": "unspecified (Ollama/model default)",
        "prompt": "existing OllamaPageInterpreter page-transcription prompt",
        "schema": "existing OllamaPageInterpreter page-transcription schema",
    }
    comparison = _comparison(run_results)
    failure_summary = {
        "ok": sum(run.status == "ok" for run in run_results),
        "invalid_response": sum(run.status == "invalid_response" for run in run_results),
        "failed": sum(run.status == "failed" for run in run_results),
    }
    result = VarianceResult(
        experiment="transcription_variance",
        runs_requested=requested,
        runs_completed=len(run_results),
        source=str(image_path),
        page_focus=prepared.focus,
        model=interpreter.model or DEFAULT_QWEN_MODEL,
        prepared_dimensions={
            "width": prepared.prepared_dimensions[0],
            "height": prepared.prepared_dimensions[1],
        },
        prepared_image_hash=image_hash,
        request_controls=controls,
        runs=run_results,
        comparison=comparison,
        failure_summary=failure_summary,
    )
    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return result


def format_variance_result(result: VarianceResult) -> str:
    lines = [
        f"experiment: {result.experiment}",
        f"source: {result.source}",
        "prepared image: "
        f"{result.prepared_dimensions['width']}x{result.prepared_dimensions['height']}",
        f"prepared image hash: {result.prepared_image_hash}",
        "",
    ]
    for run in result.runs:
        lines.append(f"Run {run.run} [{run.status}]:")
        lines.append(run.text if run.text is not None else f"{run.error or 'no transcription'}")
        lines.append("")
    if result.comparison["transcriptions_identical"]:
        lines.append("Successful parsed transcriptions are identical.")
    elif result.comparison["transcriptions_different"]:
        lines.append("Successful parsed transcriptions are different:")
        for output in result.comparison["outputs"]:
            lines.extend(["---", output])
    else:
        lines.append("No successful parsed transcriptions were available for comparison.")
    lines.append(f"failure summary: {json.dumps(result.failure_summary, sort_keys=True)}")
    return "\n".join(lines)

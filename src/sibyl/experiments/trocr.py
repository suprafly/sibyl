"""Run the local TrOCR handwritten-line experiment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

MODEL_ID = "microsoft/trocr-large-handwritten"


def local_model_snapshot() -> Path | None:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = cache_root / "hub" / "models--microsoft--trocr-large-handwritten" / "snapshots"
    required = (
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    if not snapshots.exists():
        return None
    return next(
        (
            snapshot
            for snapshot in sorted(snapshots.iterdir())
            if all((snapshot / name).exists() for name in required)
        ),
        None,
    )


class Recognizer(Protocol):
    """The small provider seam used by the experiment and its tests."""

    def recognize(self, image: Image.Image) -> tuple[str, float]: ...


@dataclass(frozen=True)
class ExperimentResult:
    source_image: str
    model: str
    device: str
    cuda_available: bool
    gpu: str | None
    model_load_ms: float
    preprocessing_ms: float
    inference_ms: float
    decoding_ms: float
    elapsed_ms: float
    text: str


class TrocrRecognizer:
    """TrOCR provider loaded only from the existing local Hugging Face cache."""

    def __init__(self, processor: Any, model: Any, device: Any) -> None:
        self._processor = processor
        self._model = model
        self._device = device

    @classmethod
    def from_local_cache(cls) -> tuple[TrocrRecognizer, float, bool, str, str | None]:
        try:
            import torch
            from transformers import (
                RobertaTokenizerFast,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                "ML dependencies are unavailable; run `just setup` first."
            ) from error

        cuda_available = bool(torch.cuda.is_available())
        device_name = "cuda" if cuda_available else "cpu"
        device = torch.device(device_name)
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
        started = time.perf_counter()
        snapshot = local_model_snapshot()
        if snapshot is None:
            raise RuntimeError(
                f"TrOCR assets for {MODEL_ID} are not available in the local Hugging Face cache. "
                "Run `just bootstrap-trocr` explicitly, then retry."
            )
        try:
            image_processor = ViTImageProcessor.from_pretrained(snapshot, local_files_only=True)
            tokenizer = RobertaTokenizerFast.from_pretrained(snapshot, local_files_only=True)
            processor = TrOCRProcessor(image_processor, tokenizer)  # type: ignore[no-untyped-call]
            model = VisionEncoderDecoderModel.from_pretrained(  # type: ignore[no-untyped-call]
                snapshot, local_files_only=True
            )
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"TrOCR assets for {MODEL_ID} are not available in the local Hugging Face cache. "
                "Run `just bootstrap-trocr` explicitly, then retry."
            ) from error
        model.to(device)
        model.eval()
        load_ms = (time.perf_counter() - started) * 1000
        return cls(processor, model, device), load_ms, cuda_available, device_name, gpu

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        import torch

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(pixel_values)
        inference_ms = (time.perf_counter() - started) * 1000
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text, inference_ms


def run_experiment(
    image_path: Path,
    recognizer: Recognizer | None = None,
    *,
    model_load_ms: float | None = None,
    cuda_available: bool | None = None,
    device: str | None = None,
    gpu: str | None = None,
) -> ExperimentResult:
    """Recognize one line image without changing the source file."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    preprocessing_started = time.perf_counter()
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unable to read image: {image_path}") from error
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000

    if recognizer is None:
        recognizer, model_load_ms, cuda_available, device, gpu = TrocrRecognizer.from_local_cache()
    if model_load_ms is None or cuda_available is None or device is None:
        raise ValueError("Runtime metadata is required when supplying a recognizer in tests.")

    decoding_started = time.perf_counter()
    text, inference_ms = recognizer.recognize(image)
    decoding_ms = (time.perf_counter() - decoding_started) * 1000 - inference_ms
    elapsed_ms = model_load_ms + preprocessing_ms + inference_ms + max(decoding_ms, 0)
    return ExperimentResult(
        source_image=str(image_path),
        model=MODEL_ID,
        device=device,
        cuda_available=cuda_available,
        gpu=gpu,
        model_load_ms=round(model_load_ms, 3),
        preprocessing_ms=round(preprocessing_ms, 3),
        inference_ms=round(inference_ms, 3),
        decoding_ms=round(max(decoding_ms, 0), 3),
        elapsed_ms=round(elapsed_ms, 3),
        text=text,
    )


def format_result(result: ExperimentResult, as_json: bool) -> str:
    if as_json:
        return json.dumps(asdict(result), indent=2)
    lines = [
        f"source image: {result.source_image}",
        f"model: {result.model}",
        f"device: {result.device}",
        f"CUDA available: {result.cuda_available}",
        f"GPU: {result.gpu or 'none'}",
        f"inference: {result.inference_ms:.3f} ms",
        f"text: {result.text}",
    ]
    return "\n".join(lines)

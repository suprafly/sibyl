"""Report the local ML environment without downloading model assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

MODEL_ID = "microsoft/trocr-large-handwritten"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def model_cache_path() -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return cache_root / "hub" / "models--microsoft--trocr-large-handwritten"


def find_local_snapshot() -> Path | None:
    snapshot_root = model_cache_path() / "snapshots"
    required = (
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    if not snapshot_root.exists():
        return None
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    return next(
        (
            snapshot
            for snapshot in snapshots
            if all((snapshot / name).exists() for name in required)
        ),
        None,
    )


def report_model_cache() -> Path | None:
    snapshot = find_local_snapshot()
    complete = snapshot is not None
    print(f"TrOCR model cached: {'yes' if complete else 'no'}")
    print(f"TrOCR cache path: {model_cache_path()}")
    return snapshot


def report_environment(load_model: bool) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    for name in ("torch", "torchvision", "transformers", "Pillow", "sentencepiece", "accelerate"):
        print(f"{name}: {package_version(name)}")

    try:
        import torch

        print(f"CUDA available: {'yes' if torch.cuda.is_available() else 'no'}")
        print(f"CUDA runtime: {torch.version.cuda or 'none'}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        else:
            print("GPU: none detected; CPU execution remains available")
    except Exception as error:
        print(f"PyTorch check: failed ({error})")
        return 1

    try:
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        print(
            "TrOCR classes importable: yes "
            f"({AutoProcessor.__name__}, {VisionEncoderDecoderModel.__name__})"
        )
    except Exception as error:
        print(f"TrOCR classes importable: no ({error})")
        return 1

    snapshot = report_model_cache()
    if load_model:
        if snapshot is None:
            print("TrOCR local load: skipped; model assets are not complete in the local cache")
            return 2
        try:
            from transformers import AutoProcessor, VisionEncoderDecoderModel

            AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            VisionEncoderDecoderModel.from_pretrained(snapshot, local_files_only=True)
            print("TrOCR local load: success")
        except Exception as error:
            print(f"TrOCR local load: failed ({type(error).__name__}: {error})")
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true", help="load TrOCR from the local cache")
    arguments = parser.parse_args()
    return report_environment(arguments.load_model)


if __name__ == "__main__":
    raise SystemExit(main())

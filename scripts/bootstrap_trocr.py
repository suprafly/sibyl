"""Explicitly download the initial TrOCR processor and model assets."""

import sys

try:
    from transformers import AutoProcessor, VisionEncoderDecoderModel
except ImportError as error:
    print("Transformers is unavailable; run `just setup` first.", file=sys.stderr)
    raise SystemExit(1) from error

MODEL_ID = "microsoft/trocr-large-handwritten"


def main() -> int:
    print(f"Downloading processor and model assets for {MODEL_ID}...")
    AutoProcessor.from_pretrained(MODEL_ID)
    VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    print("TrOCR assets downloaded to the Hugging Face cache.")
    print("Model weights remain machine-local and are not part of this repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

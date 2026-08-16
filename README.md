# Sibyl

Sibyl is a local-first Python system for transforming handwritten source
imagery into structured digital artifacts. Original scans and photographs
remain authoritative. The transform preserves wording, visual material, spatial
relationships, provenance, and explicit uncertainty; Markdown/Obsidian and
JSON are downstream projections, never the canonical artifact.

The repository includes a narrow TrOCR experiment for measuring one handwritten
line or manually selected line crop. It is not whole-page OCR.

## One-page transform

Run one page to a provider-independent JSON artifact with the local Qwen3-VL
Ollama service followed by cropped-region TrOCR recognition:

```sh
just run samples/Grafting-101-page-003.png
```

The source image remains authoritative. Qwen receives an in-memory grayscale
derivative (capped at 1536×2048), returns ordered page-level text and spatial
drawing boxes, and Sibyl crops drawings from the original image. Page-level
text does not trigger TrOCR; the spatial-text recognition boundary remains
available only for a future explicitly supported Qwen response. Structured
Qwen output is required; invalid or prose-only responses fail visibly rather
than being parsed as if they were reliable coordinates.

## Development

Requirements: Python 3.13+ and `uv`.

```sh
just venv
just test
just lint
just build
just --version
```

See [the architecture and transform vocabulary](docs/architecture.md).

See [development environment setup](docs/development.md) for reproducible
Python/ML dependencies, CUDA verification, and explicit TrOCR bootstrap.

## TrOCR experiment

Run the first empirical handwriting test with a representative real image:

```sh
just run experiment trocr path/to/one-line-image.png
just run experiment trocr path/to/one-line-image.png --json
```

The command loads `microsoft/trocr-large-handwritten` from the local Hugging
Face cache, uses CUDA when available, and falls back to CPU. Missing model
assets are reported with the explicit `just bootstrap-trocr` remedy; the
experiment does not silently download weights. The source image is converted
to RGB in memory and is never modified. Results include runtime information
and measured model-load, preprocessing, inference, decoding, and total times;
no confidence is claimed.

Use ordinary representative handwriting, including messy writing, shorthand,
neologisms, glossary-known words, and words unknown to a conventional spell
checker as the corpus grows. Do not use only specially prepared handwriting.

# Sibyl

Sibyl is a local-first Python system for faithfully recovering handwritten
material into structured digital artifacts. Original scans and photographs
remain authoritative. Recovery preserves wording, visual material, spatial
relationships, provenance, and explicit uncertainty; Markdown/Obsidian and
JSON are downstream projections, never the canonical artifact.

The repository includes a narrow TrOCR experiment for measuring one handwritten
line or manually selected line crop. It is not whole-page OCR.

## One-page recovery

Recover one page to a provider-independent JSON artifact with the local Qwen3-VL
Ollama service followed by cropped-region TrOCR recognition:

```sh
just run recover samples/Grafting-101-page-003.png
```

The source image remains authoritative. Qwen receives an in-memory grayscale
derivative (capped at 1536×2048), identifies regions and page structure, and
TrOCR recognizes each crop from the original image. Qwen is requested with
`keep_alive=0` before TrOCR loads, so the command does not assume both models
fit in the approximately 8 GB GPU simultaneously. Structured Qwen output is
required; invalid or prose-only responses fail visibly rather than being parsed
as if they were reliable coordinates.

## Development

Requirements: Python 3.13+ and `uv`.

```sh
just venv
just test
just lint
just build
just run --version
```

See [the architecture and recovery vocabulary](docs/architecture.md).

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

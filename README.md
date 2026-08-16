# Sibyl

Sibyl is a local-first Python system for faithfully recovering handwritten
material into structured digital artifacts. Original scans and photographs
remain authoritative. Recovery preserves wording, visual material, spatial
relationships, provenance, and explicit uncertainty; Markdown/Obsidian and
JSON are downstream projections, never the canonical artifact.

The current repository contains only the Python package and minimal CLI seam.
Handwriting recognition, OCR, VLM inference, model adapters, glossary
extraction, and Obsidian ingestion are intentionally not implemented.

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

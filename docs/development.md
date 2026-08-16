# Development environment

Sibyl supports Python 3.14 in the current development environment and
requires Python 3.13 or newer. `uv` is the project package manager. The
canonical dependency manifest is `pyproject.toml`; `uv.lock` records the
resolved environment.

The current experiment pins PyTorch 2.13.0, TorchVision 0.28.0, Transformers
4.57.6, Pillow 12.3.0, SentencePiece 0.2.2, and Accelerate 1.14.0. Transformers
is intentionally kept on the 4.x line for the current TrOCR checkpoint.

## Setup

```sh
just setup
```

This installs the pinned Python/ML dependencies into the project-local
`.venv`. It does not download model weights.

## Verify the environment

```sh
just check-ml
```

This reports Python, PyTorch, CUDA runtime and availability, GPU names when
present, Transformers, Pillow, SentencePiece, and Accelerate versions. It also
checks whether the TrOCR classes are importable and whether the required model
assets appear complete in the local Hugging Face cache.

CUDA and a GPU are optional. A CUDA-enabled PyTorch installation can still
run on a CPU-only machine; verification reports CUDA as unavailable instead
of treating that as a setup failure.

To perform an actual processor/model load from already-cached assets:

```sh
just check-ml-load
```

The load check fails clearly when assets are incomplete or incompatible. It
never downloads them implicitly.

## Bootstrap TrOCR explicitly

```sh
just bootstrap-trocr
```

This downloads `microsoft/trocr-large-handwritten` using the project
environment. The model cache is external and machine-local; model weights must
not be committed. Run `just run experiment trocr path/to/line.png` for the
single-line empirical experiment; it never downloads weights implicitly.

Ollama and Qwen/VLM runtimes are separate optional future boundaries. They are
not Python dependencies and are not installed or downloaded by Sibyl setup.

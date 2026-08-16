venv:
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv sync

setup: venv

check-ml: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run python scripts/verify_ml.py

check-ml-load: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run python scripts/verify_ml.py --load-model

bootstrap-trocr: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run python scripts/bootstrap_trocr.py

build: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv build

test: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run pytest

format: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run ruff format

lint: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run ruff check
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run mypy

check: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run python -m compileall -q src tests
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run ruff check
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run mypy

run *ARGS: venv
    UV_CACHE_DIR=/tmp/sibyl-uv-cache uv run sibyl {{ARGS}}

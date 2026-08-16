# Qwen3-VL benchmark recovery extension

Update the existing `sibyl recover IMAGE` implementation in place using the
first real Qwen3-VL figure-extraction benchmark. Add deterministic VLM image
preparation capped at 1536x2048 while preserving the source, prepared-to-source
coordinate mapping, Ollama/Qwen structured JSON handling from either content or
thinking with explicit structured failures, original-resolution figure crops,
single-load TrOCR recognition with per-region timing, reproducible benchmark
metadata, and deterministic unit tests with mocked model boundaries. Run both
Grafting-101 page specimens and the complete validation suite including Sextant
check/audit and git diff --check. Do not redesign recovery, add the real
samples to CI, or create/modify an ADR.

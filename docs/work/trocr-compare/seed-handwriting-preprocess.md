# Sibyl: add a controlled handwriting crop preprocessing experiment

Implement `amendments/001-handwriting-preprocess.md` as an experimental
recognition benchmark. Reuse the existing source-image crop and recognizer
machinery; do not train, alter canonical Sibyl, alter convergence, add an LLM
adjudicator, or run real model inference during implementation. The benchmark
must produce deterministic preprocessing variants, run Qwen and TrOCR over
each selected crop/variant with five reads by default, preserve raw responses
and failures, compare readings across variants/models, and keep human review
separate from recognition evidence.

Expose generic CLI targeting for existing regions and lines (line selection
precedence), optional source crop paths, `--runs`, and an auditable JSON output
plus generated crop images. Add tests, documentation, and repository
validation. Do not hard-code specimen IDs, page conditions, or expected words.

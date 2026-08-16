# Seed

Implement the requested `sibyl experiment trocr-compare IMAGE` experiment.

Compare Qwen and standard off-the-shelf handwritten TrOCR on the exact same
persisted original-RGB source-resolution coarse crops produced by the existing
region-first transcription reread localization. Default to all accepted
regions and five reads per recognizer; support `--regions` and `--runs`.
Reuse the existing regional Qwen prompt and controls (`num_predict=256`,
`think=false`, `stream=false`, `keep_alive=0`). Use the existing local TrOCR
boundary and record its model and preprocessing metadata. Never train, select
a winner, infer truth, use line localization, or change canonical Sibyl,
transcription-reread, or localization behavior.

Persist crop provenance including source bbox, padding, dimensions, RGB mode,
and SHA-256, and prove both branches use the same hash. Preserve raw Qwen
responses, parsed text, timings, TrOCR output/timings, failures, model IDs,
and comparison summaries including distinct readings and overlap. Add mocked
tests for crop identity, both adapters, failures, repeated reads,
comparisons, and isolation, plus `docs/trocr-compare.md`. Do not run real
model inference. Validate with the repository test, lint, check, Sextant, and
diff-check commands when available.

# Sibyl: visual-exemplar handwriting recognition experiment

Add an experimental benchmark named `sibyl experiment handwriting-exemplars IMAGE` to test whether
Qwen can recognize ambiguous handwriting better when shown actual visual examples from the same
writer. Use `samples/Grafting-101-page-004.png` as the specimen, but keep the implementation generic.
Do not train, add LoRA, dictionaries, hard-coded expected answers, spell correction, semantic
similarity, LLM adjudication, canonical `sibyl run` changes, or convergence changes. Do not run real
model inference.

Before implementation, materialize this seed and run `sextant prompt assemble` against it; follow
the assembled prompt. Reuse existing source localization/crop machinery. Support region, line, and
explicit target-crop targeting with line precedence, preserving source coordinates, dimensions, RGB
pixels, and hashes.

The benchmark compares target-only baseline against target plus explicitly human-confirmed visual
reference crops. A reference records id, crop path, source coordinates, hash, transcription, and
confirmed status. Reference transcription is provenance/ground truth metadata only and must never be
sent to the model. Support explicit `--references` selection and a JSON/YAML manifest; require
confirmed references, reject duplicates and target/reference identity, and order deterministically.
Generate manageable deterministic reference-set variants: no references, and prefixes of 1, 3, and 5
selected references when available. Do not invent glyph coordinates or use semantic retrieval.

Use the existing isolated handwriting prompt for baseline. For exemplar requests explicitly label
REFERENCE IMAGES and TARGET IMAGE and instruct the model to use references only for writer style and
glyph forms, transcribe only the target, not copy reference words, return only target transcription,
and not describe or infer. Never include reference or target transcriptions in the model prompt.
Keep existing baseline Qwen controls fixed; do not combine with decoding sweeps. Default to five runs
(`--runs N`) and preserve model, prompt, reference IDs, target identity, controls, raw responses,
parsed readings, statuses, timings, token counts, invalid/truncated failures, and deterministic
candidate distributions (readings, normalized readings, distinct readings, frequency, stability,
invalid/truncated counts). Optional confirmed target ground truth may produce normalized exact match,
edit distance, and token overlap, but majority is never correctness.

Write `.sibyl/experiments/handwriting-exemplars.json` and generated images under
`.sibyl/experiments/handwriting-exemplars/`. Add useful CLI help, documentation explaining visual
references versus dictionaries, leakage prevention, baseline/exemplar comparison, candidate
interpretation, and future adaptation. Add mocked deterministic tests for target selection/provenance,
manifest validation/order/leakage, prompt and image ordering, transcription non-leakage, responses,
controls, analysis, integrity, unchanged canonical behavior, and deterministic artifacts. Run `just
test`, `just lint`, `just check`, `sextant check`, `sextant audit`, and `git diff --check`.

Completion must report target/reference mechanism, prompt modes, reference-set sizes, artifact,
tests, validation, no real inference, unchanged canonical behavior, and an exact manual command for
the current specimen. The question is whether visual examples of this writer's handwriting can help
Qwen resolve an ambiguous word without being told the answer.

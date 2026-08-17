# Sibyl: add BOOX all-pages stroke inspection

Implement `sibyl experiment boox-strokes --all-pages`.

Keep `--page N` unchanged, make `--page` and `--all-pages` mutually exclusive,
and have `--all-pages` process every page in the BOOX note through the existing
verified decoder. Preserve every existing per-page artifact and provenance.
Write one deterministic corpus-level JSON summary with page count, each page's
status, stroke count, point count, total strokes, total points, and failures.
Do not change decoding or recognition behavior and do not run OCR, HTR, or model
inference. Add deterministic tests, update docs/help, and validate with `just
test`, `just lint`, `just check`, `sextant check`, `sextant audit`, and `git diff
--check`.

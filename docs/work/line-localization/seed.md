# Sibyl: subdivide localized text regions into handwriting lines

Extend experimental `transcription-reread` with a second Qwen localization
pass. Preserve the existing page preparation, coarse text localization,
validation, source mapping, coarse RGB crops, and coarse Qwen observations.
For every accepted coarse region, localize individual handwritten text lines
inside the coarse source crop using relative `qwen_0_1000` boxes and a
dedicated request budget of `num_predict=256`, `think=false`, `stream=false`,
and `keep_alive=0`. Do not change sampling controls, canonical `sibyl run`,
page transcription, drawing localization, page preparation, regional OCR,
TrOCR, adjudication, or majority voting.

Strictly validate line boxes, preserve raw responses and rejection reasons,
sort generically by top then left position, map line boxes through coarse-crop
coordinates into source coordinates, and persist padded original RGB line
crops without resizing or grayscale conversion. Use the line crop as the new
Qwen recognition unit, five independent reads per line, reusing the existing
regional reader and controls. Preserve coarse crop and coarse reads as
additive evidence. Record line-localization metadata, coordinate frames,
source mapping, crops, raw responses, failures, timings, distinct readings,
and stability in the existing experiment artifact. Do not invoke drawing
localization or create figure assets.

Add deterministic mocked tests for line localization, strict validation,
truncation, ordering, composed coordinate mapping and padding, exact RGB
pixels/no resize, line read reuse and variance, coarse evidence preservation,
and isolation. Update the reread documentation. Run `just test`, `just lint`,
`just check`, `sextant check`, `sextant audit`, and `git diff --check`; do not
run real inference. Human command:

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

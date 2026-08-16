# Sibyl: region-first handwriting recognition experiment

Replace the experimental `transcription-reread` whole-page alignment approach
with a region-first experiment. Prepare the source page once, use a dedicated
minimal experimental Qwen request to localize handwritten-text regions in
`qwen_0_1000` coordinates, validate and deterministically deduplicate boxes,
map accepted boxes to source coordinates, and crop exact original RGB pixels
with modest deterministic padding. Persist the raw localization response,
source/prepared metadata, mapping provenance, rejected regions, and reusable
`region-NN.png` assets under `.sibyl/experiments/transcription-reread/`.

Run an independent minimal structured Qwen OCR request on every crop, five
times by default with a `--runs` option. Preserve every raw response and
distinguish successful, invalid, and failed observations. Report distinct
readings and stability without majority voting, adjudication, or ground truth.

Do not modify canonical `sibyl run`, its page prompt or schema, drawing
localization, page preparation, or TrOCR. Do not invoke page transcription,
whole-page token alignment, disagreement extraction, or figure extraction for
this experiment. Reuse existing preparation, Qwen request, coordinate mapping,
crop, and artifact infrastructure. Add deterministic mocked tests covering
preparation reuse, localization validation/deduplication/order, mapping,
padding, RGB pixel preservation, run independence and reuse, raw response and
failure preservation, stability/divergence, and isolation from canonical and
drawing paths. Rewrite the reread documentation to explain the new region-first
flow and explicitly state that it does not determine ground truth.

Run `just test`, `just lint`, `just check`, `sextant check`, `sextant audit`,
and `git diff --check`. Do not run real model inference. The human validation
command is:

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

The exact artifact inspection command is:

```fish
python -m json.tool .sibyl/experiments/transcription-reread.json
```

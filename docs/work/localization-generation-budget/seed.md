# Sibyl: give experimental text localization its own generation budget

Fix the experimental region-first `transcription-reread` text-localization
request. Give it explicit controls independent of page transcription, starting
with `num_predict=128`, `think=false`, `stream=false`, and `keep_alive=0`.
Keep page transcription at `num_predict=256`; do not change canonical
`sibyl run`, canonical prompts or schemas, drawing localization, page
preparation, regional transcription, TrOCR, adjudication, or sampling
experiments.

Tighten the localization prompt and minimal structured schema so the only
output is `text_regions` with exactly four numeric `bbox_2d` values in
`qwen_0_1000`, with no prose, OCR, interpretation, reasoning, or extra
coordinates. Strictly reject malformed, non-finite, out-of-range, inverted,
zero-area, and oversized bbox arrays without repairing them. Preserve raw
responses and distinguish request failure, invalid response, and truncated
responses when `done_reason=length` is available.

Add deterministic mocked tests for controls, prompt/schema, bbox validation,
the observed oversized/truncated response, valid responses, and preservation
of the existing region-first architecture. Run `just test`, `just lint`,
`just check`, `sextant check`, `sextant audit`, and `git diff --check`. Do not
run real inference. Human validation command:

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

# Revert spatial-text expansion; preserve successful page interpretation

You are Cody. The human ran the real integration test after the previous recovery-region fix.

Do not run Qwen, Ollama, TrOCR, or any other model inference.

Before implementation:
1. Materialize this seed as a real file.
2. Run `sextant prompt assemble` against that seed.
3. Treat the assembled output as the governing implementation prompt.
4. Do not bypass prompt assembly.

## Observed real-model behavior

The previous Qwen request asked for spatial text regions. Qwen failed to produce a useful result, began producing repeated regions (`Xylem` repeatedly), and terminated with `done_reason: "length"` and `eval_count: 962`. The structured JSON was incomplete and Sibyl correctly rejected it. This is model behavior, not a JSON-parser bug.

## Architectural conclusion

Do not require Qwen3-VL 8B to produce spatial text bounding boxes for the current recovery pipeline. Preserve the successful page-level interpretation boundary:

```json
{
  "page_interpretation": {
    "text": [
      "Xylem",
      "- transports mineral nutrients and water from root to stem",
      "Phloem",
      "- transports food and nutrients from leaves to storage organs.",
      "Sapling grafting - what we will do now.",
      "N -> H -> Wurd"
    ],
    "diagram": [{"bbox": [0.329, 0.717, 0.427, 0.874], "description": "..."}]
  }
}
```

A later real run demonstrated that Qwen can produce multiple drawing boxes. Preserve that successful model boundary.

## Required Qwen recovery shape

Request and normalize page interpretation with page-level `text[]` and `drawing[]` with normalized bounding boxes. Do not request every text fragment with a bounding box, ask Qwen to enumerate spatial text regions, or make recovery depend on spatial text localization.

## Text handling

Treat Qwen's page-level `text[]` as ordered page-level recovered text. Use it directly for `sibyl recover IMAGE` and `sibyl recover IMAGE --markdown`. Preserve model output exactly; do not silently correct Xylem, Phloem, Splice/Splits, wrap/Wurd, arrows, capitalization, or punctuation. Do not run TrOCR against page-level text or manufacture spatial bounds. For the current mode, `spatial_text_regions = 0` and `TrOCR_attempts = 0` unless Qwen provides legitimate spatial text regions through a future explicitly supported model response.

## Drawing handling

Continue supporting spatial drawing regions. For each drawing, read normalized `[x1,y1,x2,y2]`, map to prepared-image coordinates, map to original source-image coordinates, crop from the ORIGINAL source image, write an original-resolution asset, expose it in recovery JSON, and embed it in Markdown. The drawing description is model evidence; the original image crop is the visual source of truth.

The model may return one drawing encompassing a sequence or several drawing regions. Both are valid. Do not force a particular count; preserve the regions Qwen actually returns.

## Projection

`sibyl recover IMAGE` outputs page-level recovered text to stdout. `sibyl recover IMAGE --markdown` outputs page-level recovered text plus extracted drawing assets. `sibyl recover IMAGE --json` outputs complete structured recovery. Do not make Markdown or JSON the canonical internal representation.

## TrOCR

Do not remove the TrOCR implementation. Keep the legitimate spatial-text recognition boundary available for future model responses, but it must not execute when the current Qwen result only contains page-level text. No fake full-page regions, repeated full-page OCR, or synthetic coordinates.

## Tests

Update mocked tests so the successful page-level interpretation is the primary fixture. Test page-level text preservation; no TrOCR for page-level text; spatial drawing regions; normalized drawing coordinate mapping; one or multiple drawings producing corresponding assets; Markdown asset references; clear rejection of incomplete/length-truncated Qwen JSON; clear errors for unsupported schemas; and acceptance of valid `page_interpretation` JSON in `message.thinking`. Do not run model inference.

## Validation

Run `just test`, `just lint`, `just check`, `Sextant check`, `Sextant audit`, and `git diff --check`. Do not run the real specimen. Report implementation changes and the exact command for the next human real-model test. Do not create or modify an ADR.

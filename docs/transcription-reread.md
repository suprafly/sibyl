# Region-first transcription reread experiment

This experiment measures repeated independent Qwen observations of handwriting
line crops. It does not use page transcription, string alignment, disagreement
extraction, drawing localization, or figure cropping.

```text
OLD

whole-page transcription → string alignment → guessed uncertainty

NEW

page → coarse text regions → source RGB crops → line localization → source RGB line crops → Qwen × 5
```

The page is prepared exactly once. A dedicated, minimal spatial request finds
coarse handwritten-text boxes in Qwen's `qwen_0_1000` coordinates. Each
accepted coarse crop is then given to a second minimal line-localization
request. Its boxes use the same coordinate convention relative to the coarse
crop, are ordered top-to-bottom then left-to-right, mapped into source
coordinates, and cropped from the original RGB source pixels. Both coarse
evidence and line evidence remain in the artifact.

Each coarse crop retains its five independent reads, and each line crop
receives five independent minimal `text` requests by default. The artifact
preserves every response, parsed reading, duration, and failure. It reports
distinct successful readings and marks a unit stable only when all successful
observations agree. It never majority-votes, selects a winner, or calls one
reading correct.

Line crops reduce irrelevant visual context, make handwriting failures easier
to inspect, preserve source pixels, and provide an appropriate future TrOCR
input. The same line crop can later be passed to Qwen and TrOCR. This
experiment does not determine ground truth.

## Run it

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

Override the number of independent reads per region with `--runs`, for example
`--runs 5`. The experimental artifact is written to:

```text
.sibyl/experiments/transcription-reread.json
```

Reusable coarse and line crops are written beside it as `region-01.png` and
`region-01-line-01.png`. They are experimental assets and never enter
canonical `assets/` or the canonical transform artifact.

Inspect the artifact with:

```sh
python -m json.tool .sibyl/experiments/transcription-reread.json
```

TrOCR is deliberately not implemented in this phase. The intended later
comparison is the same source crop through Qwen × N and TrOCR × N.

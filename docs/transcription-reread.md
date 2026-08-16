# Region-first transcription reread experiment

This experiment measures repeated independent Qwen observations of the same
localized handwriting crop. It does not use page transcription, string
alignment, disagreement extraction, drawing localization, or figure cropping.

```text
OLD

whole-page transcription → string alignment → guessed uncertainty

NEW

page → text-region localization → source-resolution crops → Qwen × 5 → regional variance
```

The page is prepared exactly once. A dedicated, minimal spatial request finds
handwritten-text boxes in Qwen's `qwen_0_1000` coordinates. Valid boxes are
deterministically ordered and deduplicated, mapped through the existing
prepared-to-source mapping, and cropped from the original RGB source pixels.
The persisted crops are the canonical visual evidence and are reusable by a
future TrOCR experiment.

Each crop receives five independent minimal `text` requests by default. The
artifact preserves every response, parsed reading, duration, and failure. It
reports distinct successful readings and marks a region stable only when all
successful observations agree. It never majority-votes, selects a winner, or
calls one reading correct.

The crop is the unit of visual evidence: line wrapping cannot be confused with
handwriting disagreement, diagram interpretation does not contaminate regional
reads, and the same source pixels can later be passed to TrOCR. This
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

Reusable crops are written beside it as `region-01.png`, `region-02.png`, and
so on. They are experimental assets and never enter canonical `assets/` or the
canonical transform artifact.

Inspect the artifact with:

```sh
python -m json.tool .sibyl/experiments/transcription-reread.json
```

TrOCR is deliberately not implemented in this phase. The intended later
comparison is the same source crop through Qwen × N and TrOCR × N.

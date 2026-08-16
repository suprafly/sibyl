# Targeted transcription reread experiment

This investigation follows the transcription-variance experiment. It repeats
page observations, normalizes presentation-only variation, identifies actual
divergent text spans, and then gathers independent rereads of localized
source-resolution text crops. It measures whether targeted visual evidence
narrows uncertainty; it does not produce production OCR or ground truth.

The experiment intentionally does not majority-vote or adjudicate. Every page
candidate and every targeted reread remains an observation for later human
evaluation. Targeted rereads do not receive the competing candidate list, so
they are independent visual observations rather than prompted choices.

The comparison treats Markdown bullets, repeated blank lines, whitespace, and
line wrapping as presentation. It preserves lexical tokens, punctuation,
numbers, capitalization, and meaningful word boundaries. The conceptual flow
is:

```text
page observations
        ↓
normalized textual comparison
        ↓
actual divergent spans
        ↓
text-region localization
        ↓
source-resolution crop
        ↓
independent targeted reread
```

The result is additional evidence, not a determination of ground truth.

## Run it

Use the project-local CLI. Page observations default to five runs and targeted
rereads default to three per disagreement:

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

Override either count with `--runs` or `--rereads`. Results are written to:

```text
.sibyl/experiments/transcription-reread.json
```

The experiment prepares the page once and reuses the prepared representation
for page observations and experimental localization. It compares structured
page-text lines by their response order; it never turns a string search or a
specimen-specific coordinate into a crop.

Because the canonical page response does not provide reliable text boxes, a
separate experimental localization request asks for ordered text-line boxes
in Qwen's established `0..1000` coordinate space. Valid boxes are mapped to
source pixels with the existing coordinate helpers. Crops use original RGB
source pixels, deterministic 5% proportional padding, and are stored under:

```text
.sibyl/experiments/transcription-reread/region-NN.png
```

They never enter canonical `assets/`. If localization is invalid or does not
cover a disagreement, the artifact preserves the candidates and reports
`candidate disagreement detected` and `targeted localization unavailable`
without fabricating a crop.

Each valid disagreement crop receives three independent minimal reread
requests by default. The reread asks only for exact text in the crop, returns a
single `text` field, and does not receive competing candidates or surrounding
page context. Raw responses, parsed readings, failures, durations, coordinates,
padding, dimensions, and paths remain in the experimental JSON.

No drawing localization, figure extraction, canonical transform artifact, or
adjudication runs as part of this experiment.

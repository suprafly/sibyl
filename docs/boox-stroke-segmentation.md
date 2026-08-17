# BOOX stroke-aware segmentation

This experiment tests whether native BOOX stroke geometry improves Markdown
recovery by improving segmentation. BOOX strokes determine where candidate
text units are; recognition still operates on crops of the original raster
page image. This is different from BOOX-assisted recognition, where native
stroke renders are supplied as visual references.

The experiment compares accepted coarse regions, existing visual line crops,
BOOX-derived lines, and BOOX-derived word/phrase crops. BOOX groups use
deterministic vertical-center proximity and horizontal-gap heuristics. Their
parameters, stroke provenance, identity source mapping, crop hashes, rejected
groups, and reading order are recorded in the JSON artifact.

The verified BOOX page remains 1404×1872 native coordinates. The supplied raster
page is inspected at runtime and accepted when its aspect ratio is compatible;
native group geometry is then deterministically scaled into raster coordinates.
The artifact preserves both page dimensions, both scale factors, and native and
raster bboxes. Crops and overlays always use pixels from the supplied original
raster image.

Run the human-owned page-4 experiment with one read per crop:

```fish
uv run sibyl experiment boox-stroke-segmentation \
  samples/Grafting-101-page-004.png \
  --note "samples/Grafting 101.note" \
  --runs 1 \
  --num-predict 2048 \
  --num-ctx 8192
```

Use `--runs 3` for repeated-read selection. An optional review file contains
`{"lines": [{"text": "..."}]}` in page reading order; it is evaluation-only.
The output is `.sibyl/experiments/stroke-segmentation.json`. Markdown files,
original-raster crops, and diagnostic overlays are under
`.sibyl/experiments/stroke-segmentation/`.

The experiment preserves `⟦unresolved⟧` when the configured recognition runs
do not satisfy the existing conservative selector. It does not change
canonical Sibyl transformation or page recovery.

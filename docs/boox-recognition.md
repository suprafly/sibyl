# BOOX native-stroke-assisted handwriting recognition

`boox-recognition` is an isolated inference-time experiment. Its hypothesis is:

> Native stroke trajectories provide writer-specific visual exemplars that may
> help an image-based recognizer decode otherwise ambiguous handwriting.

It is designed to falsify that hypothesis. It does not train or fine-tune a
model, modify canonical Sibyl recognition, or send human transcriptions to
Qwen. Transcriptions are accepted only through an explicit confirmed review
file and are applied after recognition for evaluation.

The experiment uses the verified page-4 BOOX decode: 167 strokes, 17,272
points, 1404×1872 native dimensions, identity coordinates, and verified
stroke/point associations. It reuses the existing Qwen adapter and fixed
recognition controls, repeating each condition for the requested number of
reads while preserving raw and parsed responses, failures, prompts, controls,
candidate distributions, and stability.

The conditions are:

- `baseline`: target image only.
- `native-render`: target plus a deterministic full-page native-stroke render.
- `native-exemplar`: target plus one other text-line native-stroke crop.
- `multi-exemplar`: target plus all available other text-line crops.
- `leave-one-region-out`: target plus only references whose source regions do
  not intersect the target region.

Reference records contain source page, source and native bounds, exact native
stroke IDs, point counts, dimensions, rendering parameters, and hashes. No
geometric fitting is performed: native renders use identity coordinates, while
source crop selection uses the recorded page-size mapping from the page image
to the verified native page dimensions.

The default targets are the existing page-4 line IDs covering the Xylem,
Phloem, mineral-nutrients, and water-to-scion test material, plus the existing
region containing “Splice grafting - what we will do now”. Use `--lines` or
`--regions` to select existing targets; line selection takes precedence when a
selection is explicitly supplied. The existing region/line artifacts must
already be available.

Run the mocked-free experiment manually with:

```sh
uv run sibyl experiment boox-recognition \
  --lines region-02-line-01,region-02-line-02,region-02-line-03,region-03-line-01 \
  --runs 5
```

For a quick smoke test, restrict both the target and conditions:

```sh
uv run sibyl experiment boox-recognition \
  --lines region-02-line-03 \
  --conditions baseline,leave-one-region-out \
  --runs 1
```

The JSON artifact is checkpointed after each completed condition. An
interrupted run remains marked `status: "running"` and preserves completed
results; a finished run is marked `status: "complete"`.

For evaluation, provide a separate confirmed review file:

```json
{
  "targets": [
    {"target_id": "region-02-line-03", "transcription": "water from root to scion", "confirmed": true}
  ]
}
```

The artifact is `.sibyl/experiments/boox-recognition.json`; rendered native
references are under `.sibyl/experiments/boox-recognition/`. Compare the
condition-level exact match, token/word overlap, unresolved tokens, candidate
distribution, and stable reading. A result is not an improvement claim unless
the observed evaluation demonstrates it across the controlled comparison.

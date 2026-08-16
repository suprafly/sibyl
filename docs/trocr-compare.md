# Qwen vs TrOCR comparison experiment

`trocr-compare` measures two recognizers on one persisted crop:

```text
same source pixels
      │
      ├── Qwen
      │
      └── TrOCR
```

The experiment reuses the accepted coarse text regions and source-pixel crop
mapping from `transcription-reread`. Each crop is saved as original RGB PNG
without resizing or grayscale conversion before either recognizer reads it.
The artifact records the crop hash twice so the identity invariant is visible.

Qwen uses the existing regional transcription contract. TrOCR uses the local
off-the-shelf `microsoft/trocr-large-handwritten` adapter. Standard TrOCR
generation may be deterministic, so identical repeated outputs are retained as
evidence rather than randomized.

Agreement between recognizers is evidence of agreement, not proof of
correctness. The experiment does not choose a winner, establish ground truth,
or perform adjudication. Its purpose is to show whether TrOCR provides a
complementary recognition signal to Qwen, including cases where one model is
stable and the other varies.

Run it with five reads by default:

```sh
uv run sibyl experiment trocr-compare samples/Grafting-101-page-004.png
uv run sibyl experiment trocr-compare samples/Grafting-101-page-004.png \
  --regions region-01,region-02,region-03,region-04,region-05,region-10
```

The TrOCR checkpoint must already exist in the local Hugging Face cache. If it
is not present, install it explicitly with the repository's documented
`just bootstrap-trocr` command; the comparison command never downloads model
weights automatically. Results are written to
`.sibyl/experiments/trocr-compare.json` unless `--output` is supplied.

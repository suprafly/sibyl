# Handwriting preprocessing experiment

`handwriting-preprocess` measures whether deterministic input representations
change ambiguous handwriting recognition. It is an experiment over crops
already persisted by `transcription-reread` and `trocr-compare`; it does not
localize new regions or create a second coordinate system.

The source crop remains unchanged and its source bbox, dimensions, RGB mode,
and SHA-256 are recorded. The benchmark writes auditable variant images under
`.sibyl/experiments/handwriting-preprocess/<target>/` and records each variant's
dimensions, parameters, and hash.

## Variants and reads

Each target is tested in this deterministic order:

1. original RGB
2. grayscale
3. 2× RGB
4. 3× RGB
5. 2× grayscale
6. contrast-normalized grayscale
7. contrast-normalized 2× grayscale

Resize uses Pillow's deterministic Lanczos resampling. Contrast normalization
uses Pillow autocontrast. Aspect ratio is preserved and no padding or AI
enhancement is applied. Qwen and TrOCR receive every variant, with five reads
per recognizer and variant by default. `--runs N` overrides the count; the
existing `SIBYL_TRANSCRIPTION_REREAD_RUNS` environment setting is also
accepted when the API does not pass an explicit count.

Every raw Qwen response, parsed reading, invalid/truncated response, failure,
timing, distinct reading, and stability result is retained. TrOCR's existing
adapter does not expose a raw transport response, so its raw-response field is
explicitly `null` while returned text and failures remain recorded.

## Targeting and review

Region targets are read from the existing `trocr-compare` artifact. Line
targets are read from `transcription-reread`; `--lines` takes precedence over
`--regions`. An explicit `--crop` can target an already persisted crop. If
line localization is too coarse to isolate a word, the artifact preserves that
limitation rather than inventing coordinates.

Recognition candidates are evidence, not truth. Cross-variant stability,
variant-specific readings, variance, and cross-model agreement are reported
without semantic correction. Optional human review is separate:

```yaml
ground_truth:
  text: "human-confirmed reading"
  confirmed: true
```

When confirmed, the artifact reports exact match, normalized exact match,
token overlap, and character edit distance. No expected specimen word is
embedded in runtime code.

## Example

```sh
uv run sibyl experiment handwriting-preprocess \
  samples/Grafting-101-page-004.png \
  --regions region-02 \
  --runs 5
```

To test an existing line crop when available:

```sh
uv run sibyl experiment handwriting-preprocess \
  samples/Grafting-101-page-004.png \
  --regions region-02 \
  --lines region-02-line-04 \
  --runs 5
```

This experiment does not train a model, add LoRA, invoke an adjudicator, or
modify canonical `sibyl run` or document convergence. Its real-model results
must be interpreted by a human after running the command.

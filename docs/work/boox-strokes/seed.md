# Sibyl: add BOOX native stroke inspection experiment

Add a deterministic, non-OCR experiment command:

```text
sibyl experiment boox-strokes NOTE
```

It supports `--page N` and `--output DIR`, defaults to page 4, does not modify
the `.note` or PDF, and does not change canonical Sibyl behavior. Use the
verified `samples/Grafting-101-corpus.json` manifest when available for page
mapping.

The experiment must inspect the native BOOX `.note` container, recording its
format, ordered page IDs, dimensions, shape/resource and point-resource IDs,
file sizes, hashes, and reconstruction-relevant metadata. It must preserve raw
resources and mark point decoding incomplete when the binary format is not
confidently understood; it must not guess.

For a selected page, record BOOX/PDF mapping, dimensions, confidence, and
recoverable native strokes with shape/resource IDs, point-resource IDs, point
counts, coordinates, bounds, ordering, pen metadata, and source hashes. Keep
native and mapped page coordinates when both exist, with explicit uncertainty.

When confidently decodable, render deterministic `page-NNN-native.png` and
`page-NNN-strokes.png` artifacts using native dimensions, ordering, geometry,
and confidently understood pen metadata. Produce overlay/diff diagnostics when
practical. Persist deterministic hashes, bounds, counts, coverage, mapping,
raw resources, reconstruction parameters, warnings, and comparison metadata in
`.sibyl/experiments/boox-strokes/` and
`.sibyl/experiments/boox-strokes.json`.

Add mocked/unit tests for container discovery, page enumeration/order/mapping,
hashes, resource and association discovery, coordinate decoding/bounds/mapping,
ordering, reconstruction/aspect ratio/pen metadata, raw and uncertain formats,
determinism, source immutability, unchanged canonical run, and no model
inference. Do not run OCR, HTR, convergence, Qwen, TrOCR, or other model
inference.

Document the value of native strokes, known structures, mapping, limitations,
provenance, and future corpus use. State explicitly that this experiment does
not perform handwriting recognition or model training; it establishes whether
native BOOX stroke data can be recovered as a trustworthy handwriting
representation.

Validate with `just test`, `just lint`, `just check`, `sextant check`,
`sextant audit`, and `git diff --check`, using repository-equivalent commands
where needed. Provide the exact page-4 manual command and inspection commands,
and distinguish Cody validation from human real-model validation.

# BOOX native stroke inspection

Rendered PDF pixels are a record of appearance; a BOOX `.note` may also retain
the original ordered pen samples, shape metadata, and page geometry. Native
strokes are valuable because they can become a spatially faithful handwriting
corpus without treating rasterization artifacts as handwriting evidence.

The experiment reads the NOTE ZIP container without changing it. The current
corpus exposes `note/pb/note_info`, ordered page IDs and 1404×1872 dimensions,
per-page `shape` resources, and per-page `point` resources. Shape resources
contain protobuf-like records and JSON fragments; point files have stable
resource IDs and raw hashes. The point binary encoding is intentionally only
decoded when every record passes conservative page-bound checks. Otherwise the
raw resource is preserved and the metadata says decoding is incomplete.

Each run also writes a page forensic dump and byte-for-byte raw resources. On
the verified page-4 specimen, the shape entry is a ZIP containing a complete
wire-parseable stream with 167 repeated field-1 records. Its observed nested
fields include the point-resource ID, shape ID, timestamps, JSON bounds, and
pen metadata. The point resource has a fixed prefix containing the page ID and
point-resource ID, followed by a repeated 16-byte-aligned region. Big-endian
float pairs occur in plausible page ranges, but multiple record layouts and
stroke boundaries remain unresolved; no coordinate hypothesis is accepted by
the experiment.

Run the page-4 experiment with:

```sh
uv run sibyl experiment boox-strokes \
  "samples/Grafting 101.note" --page 4 \
  --output .sibyl/experiments/boox-strokes
```

Inspect the JSON with `jq . .sibyl/experiments/boox-strokes.json` and
`jq . .sibyl/experiments/boox-strokes/page-004-metadata.json`; view
`page-004-native.png`, `page-004-strokes.png`, and any overlay/diff PNG in an
image viewer.

The forensic files are `page-004-forensics.json`,
`page-004-point-resource.bin`, and `page-004-shape-resource.bin`. The JSON
distinguishes wire-format observations from semantic coordinate hypotheses.

The corpus manifest is used for page mapping when its ordered page IDs confirm
the selected page. Any coordinate transform is recorded explicitly. Rendering
uses native dimensions and deterministic geometry; pen appearance is marked
approximate unless its metadata is understood. A blank or partial rendering is
evidence of incomplete decoding, not a claim that the source page is blank.

This experiment does not perform handwriting recognition or model training. It
establishes whether native BOOX stroke data can be recovered as a trustworthy
handwriting representation. Cody's tests are mocked/deterministic; a human
must inspect real specimen artifacts and assess whether major handwriting
structures align with the rendered page.

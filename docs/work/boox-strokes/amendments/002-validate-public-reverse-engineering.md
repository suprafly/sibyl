# Amendment: validate public BOOX reverse-engineering work

Before continuing format work from scratch, investigate and document the
public `boox-note-dump` discussion, `boox-note-parser` Rust crate, and related
`boox-note-optimizer` tooling. Treat all external schemas as hypotheses until
they decode the actual page-4 resources from `samples/Grafting 101.note`.

Add a narrowly scoped experimental probe that compares the documented point
file/header/table/stroke layout, IDs, packing, byte order, and coordinate
representation against the preserved page-4 forensic artifacts. It must report
external schema matched, partially matched, or does not match, with offsets,
expected/observed bytes, decoded values, ranges, and confidence. If it decodes
points, validate ordering, strokes, associations, 1404x1872 bounds, transforms,
and deterministic reconstruction. If it fails, record the exact divergence.

Preserve the forensic parser and raw resources, do not replace canonical
behavior, do not run OCR/HTR/model inference/training, and do not force an
external schema. Update BOOX documentation with direct attribution links and
distinguish external documentation, specimen observations, hypotheses, and
verified findings. Add tiny deterministic fixtures for compatible headers,
tables, strokes, coordinates, malformed/wrong-version resources, associations,
bounds, and deterministic reconstruction. Validate with the repository test,
lint, check, Sextant, audit, and diff-check commands.

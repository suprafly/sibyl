# Sibyl: validate BOOX point decoding against existing reverse-engineering work

Investigate the public `boox-note-dump` discussion at
https://www.reddit.com/r/Onyx_Boox/comments/1fva066, the `boox-note-parser` Rust
crate documentation at https://docs.rs/boox-note-parser/latest/boox_note_parser/,
and related tooling at https://github.com/nrontsis/boox-note-optimizer.

Use only page 4 of `samples/Grafting 101.note` and the existing forensic
artifacts. Treat external schemas as hypotheses until they successfully decode
the actual bytes. Preserve raw resources and the existing conservative decoder.

Study and record the external point-file header, point table, stroke structure,
coordinates, field widths, byte order, packing/compression, IDs, associations,
and version/device assumptions, distinguishing documented facts from inference.
Add an isolated experimental probe that reports external schema matched,
partially matched, or does not match, with offsets, expected and observed bytes,
decoded values, ranges, and confidence. Compare headers, tables, records,
resource IDs, alignment, and coordinate encoding against page 4.

If points decode, validate point/stroke counts, ordering, pressure/timestamps,
pen metadata, shape-to-point association, 1404x1872 bounds, coordinate origin,
units, transforms, deterministic reconstruction, and spatial correspondence to
the PDF without arbitrary fitting. Integrate only as a separated experimental
layer if verified; otherwise report exact divergence and use it as evidence for
the next investigation. Do not run OCR, HTR, Qwen, TrOCR, convergence, or
training.

Add deterministic tiny-fixture tests for compatible headers, point tables,
strokes, coordinates, malformed/truncated and wrong-version resources, ID
association, bounds, and deterministic reconstruction. Update BOOX docs with
direct external links and attribution. Run `just test`, `just lint`, `just
check`, `sextant check`, `sextant audit`, and `git diff --check`.

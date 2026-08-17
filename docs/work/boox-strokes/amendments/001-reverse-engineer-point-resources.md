# Amendment: reverse-engineer BOOX native point resources

Investigate only page 4 of `samples/Grafting 101.note` scientifically. Preserve
the note, raw point/shape bytes, hashes, and sizes. Add forensic artifacts
`page-004-forensics.json`, `page-004-point-resource.bin`, and
`page-004-shape-resource.bin` under the BOOX experiment output.

Add a diagnostic wire-format parser that tests protobuf varint, fixed32,
fixed64, and length-delimited fields, nested messages, offsets, lengths, raw
hex, and repeated structures without assigning uncertain fields semantic names.
Investigate shape-to-point association and metadata clues. Only produce
candidate coordinate renderings when a hypothesis yields substantial ordered
points, coherent bounds within 1404x1872, correct association, and spatial
correspondence with PDF corpus page 4. Candidates must be separate from the
existing conservative native rendering and record fields, transforms, counts,
bounds, and confidence explicitly.

Do not guess, modify the existing conservative decoder, process all pages, run
OCR/HTR/model inference, or train anything. Add deterministic tests separating
wire parsing from semantic interpretation, including nested/malformed input,
raw-byte preservation, candidate extraction/bounds, and deterministic rendering.
Run the repository test, lint, check, Sextant, audit, and diff-check commands.
Do not claim success unless the actual page-4 bytes satisfy the complete
coordinate/reconstruction criterion; otherwise report structural findings and
uncertainty.

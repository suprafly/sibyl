# Sibyl: reverse-engineer BOOX native point resources

Investigate only page 4 of `samples/Grafting 101.note` scientifically. Do not
modify the existing conservative decoder, run OCR or HTR, train anything, or
process all 13 pages.

Preserve the source note, raw page-4 point and shape bytes, hashes, and sizes.
Produce `page-004-forensics.json`, `page-004-point-resource.bin`, and
`page-004-shape-resource.bin` under the BOOX experiment output. The forensic
dump must expose offsets, lengths, confidently identifiable protobuf-like field
numbers/wire types, varints, fixed-width values, length-delimited boundaries,
nested messages, repeated structures, and raw hex around candidates. Determine
whether the bytes are protobuf or merely protobuf-like without requiring a
schema and without labeling unknown fields as coordinates.

Investigate repeated point structures, candidate x/y/pressure/time fields,
page bounds (1404x1872), ordering, shape-to-point association, pen metadata,
timestamps, compression, quantization, and coordinate origin. Record observed
facts separately from hypotheses and external knowledge. Candidate renderings
must be separate from `page-004-native.png`, explicitly record fields,
transforms, scale, offset, counts, bounds, confidence, and deterministic PDF
geometry diagnostics. Do not accept a candidate unless it produces substantial
ordered points, coherent in-page geometry, correct resource association, and
spatial correspondence to PDF corpus page 4.

Add deterministic tests for wire parsing, nested messages, field preservation,
malformed/truncated input, raw-byte preservation, candidate extraction/bounds,
and deterministic rendering. Tests must not encode a guessed BOOX schema as
truth. Run `just test`, `just lint`, `just check`, `sextant check`, `sextant
audit`, and `git diff --check`. Report unresolved uncertainty if the actual
bytes cannot be decoded confidently, and do not claim model validation.

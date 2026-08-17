# Sibyl: deterministic BOOX corpus PDF preparation

Add a small local utility/CLI command that creates a reduced PDF for the pages represented by the
BOOX native note in `samples/Grafting 101.note` (the workspace file has no `(1)` suffix) and the full
`samples/Grafting 101.pdf`. Do not modify either source. Do not implement handwriting recognition,
training, model changes, or a general `.note` parser.

Use the simplest existing PDF tooling available, preserve original PDF pages with a page-copy
operation rather than re-rendering/recomposing, and write `samples/Grafting-101-corpus.pdf` plus a
small provenance manifest. The manifest records source PDF, source note, selected one-based source
page numbers, output PDF, page count, SHA-256 hashes, page dimensions, and mapping assumptions.

Determine the mapping from available note page metadata, dimensions, and ordering. If confident, the
current evidence is the note's 13 ordered page IDs and 1404x1872 page dimensions matching the PDF's
13 pages and 1404x1872-point page size, so record the identity mapping assumption. If count or
dimensions do not match, fail clearly without selecting pages. The command must be deterministic,
must not overwrite source files, and must provide useful help and an exact provenance inspection
command. Add mocked/unit tests for metadata matching, mismatch refusal, page-copy preservation,
hashes, manifest fields, and deterministic output. Run the full repository validation suite and do
not run real model inference.

# BOOX corpus PDF preparation

`corpus boox-reduce` prepares a reduced PDF without changing the native BOOX
note or the full source PDF. It reads only the note's page order and page
dimensions, then verifies that the PDF has the same page count and page size.
The current Grafting 101 sources establish an identity mapping: 13 ordered
note pages, each 1404×1872, match 13 PDF pages at 1404×1872 points.

The command refuses to select pages when count or dimensions disagree. It uses
Poppler's `pdfseparate` and `pdfunite` to copy PDF pages without rasterizing or
recomposing them. A JSON manifest records source/output paths and hashes,
selected one-based pages, dimensions, note page IDs, and the mapping
assumptions.

```sh
uv run sibyl corpus boox-reduce \
  "samples/Grafting 101.note" \
  "samples/Grafting 101.pdf" \
  --output samples/Grafting-101-corpus.pdf \
  --manifest samples/Grafting-101-corpus.json
```

Inspect the provenance with:

```sh
jq . samples/Grafting-101-corpus.json
```

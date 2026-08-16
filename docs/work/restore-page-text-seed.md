# Sibyl: restore page-level text transformation and retain single figure extraction

Restore page-level Qwen handwriting transformation because the spatial-text to
TrOCR experiment produced materially worse transcription. Remove the active
spatial text localization, text crops, TrOCR canonical text, TrOCR metrics, and
spatial ordering production path. Do not leave dead production architecture.

Keep the dedicated drawing-localization pipeline exactly unchanged: its prompt,
schema, Qwen 0..1000 bbox semantics, normalized/prepared/source mapping,
padding, original-resolution crop, provenance, Markdown projection, and single
complete figure asset. A normal run must produce at most the one complete
`assets/figure-01.png`, never diagnostic `text-*.png` assets.

The production flow is page-level Qwen transformation for page text plus the
dedicated Qwen drawing localizer for one complete figure crop. Page Qwen owns
ordinary handwritten notes, headings, bullets, textual lines, and unfamiliar
terminology. It must exclude obvious graphical arrows, diagram strokes,
connectors, and purely graphical marks without excluding handwritten words near
drawings.

Use a concise page-level prompt that asks Qwen to transcribe handwritten notes
in reading order, preserve wording/spelling/capitalization/punctuation/shorthand
and unfamiliar terminology, avoid invention or semantic correction, use
`[unclear]` only for genuinely unreadable handwriting, exclude graphical diagram
marks, retain nearby handwritten words, and return only structured JSON. Do not
include specimen-specific vocabulary or examples.

Reuse the existing page-level structured response model and preserve Qwen page
text as canonical page text. Keep transform.json, transform.md, and figure
assets. Remove misleading normal-transform fields that only describe spatial
text/TrOCR, including text localization and TrOCR timing/attempt/success/failure
metrics and disagreement data. Keep page and drawing timing/provenance.

Update mocked tests to cover page-level structured text, faithful wording,
graphical exclusion and nearby handwriting prompt contracts, no specimen words,
one complete figure bbox/crop/asset, JSON/Markdown projections, and absence of
text diagnostic assets. Preserve all drawing coordinate and crop regressions.

Do not add OCR, TrOCR, spatial text localization, text-region crops, classical
CV, semantic correction, vocabulary correction, specimen rules, another VLM, or
image-preparation/geometry changes. Do not modify an ADR. Do not run Ollama,
Qwen, Qwen3-VL, TrOCR, `sibyl run` on a real specimen, or any other inference.

Validate with `just test`, `just lint`, `just check`, `sextant check`,
`sextant audit`, and `git diff --check`. The human-only integration command is
`just run samples/Grafting-101-page-004.png --markdown`.

# Sibyl: simplify spatial text localization and complete the transform pipeline

Implement the smallest integration that composes the already-working Qwen page
understanding, Qwen drawing localization, Qwen 0..1000 coordinate mapping,
original-resolution crops, persisted TrOCR crops, and TrOCR provenance.

Do not run Ollama, Qwen, Qwen3-VL, TrOCR, `sibyl run` on a real specimen, or
any other model inference. Use mocked responses and deterministic fixtures.

## Required architecture

Qwen page/text localization only answers where handwritten text regions are.
Its complete response schema is:

```json
{"text_regions":[{"bbox_2d":[x1,y1,x2,y2]}]}
```

`bbox_2d` is Qwen3-VL's explicit 0..1000 coordinate space. Reuse the existing
conversion through normalized coordinates, the prepared image, and the
prepared-to-source mapping. Do not create another coordinate representation.

Use this short prompt, without specimen-specific words or added structured
fields:

Identify every distinct region of handwritten textual notes on this page.

Return one bounding box for each coherent text block.

Do not include arrows, diagram strokes, drawing lines, graphical connectors,
or other purely graphical elements.

Handwritten words remain text even when they are near a drawing.

Return only the requested JSON.

Do not request order, kind, text, label, description, page interpretation,
coordinate explanations, reading order, transcription, or reasoning.

Sibyl assigns deterministic reading order from page coordinates: approximately
top-to-bottom, then left-to-right within a line/block, using existing geometry
utilities where available. Qwen returns no order and no transcription.

For each region, reuse the existing deterministic text padding, clamping,
prepared-to-source mapping, original-resolution crop, persisted `text-NN.png`
asset, and crop provenance. Send exactly that source crop to the existing
TrOCR boundary. TrOCR output is canonical text; do not add correction,
vocabulary logic, or another OCR engine.

For zero regions, record zero spatial regions and zero TrOCR attempts. Never
fabricate a full-page region. If localization fails, preserve the failure and
do not run TrOCR. If one TrOCR region fails, preserve the region and failure,
continue other regions, and never manufacture text.

Preserve per-region order, raw Qwen bbox, coordinate space, prepared bounds,
source bounds, padding, crop path and dimensions, TrOCR status/timing/text, raw
Qwen response, and available page-level Qwen evidence. Successful TrOCR text
is canonical. Keep the existing transform.json, transform.md, and assets/
artifact model; Markdown uses canonical TrOCR text and figure assets, not
diagnostic text crops.

Keep metrics preparation_ms, page_transform_ms, drawing_localization_ms,
text_localization_ms, trocr_ms, crop_ms, total_transform_ms,
spatial_text_regions, trocr_attempts, trocr_successes, and trocr_failures.

Freeze the drawing prompt, schema, Qwen 0..1000 semantics, conversion, padding,
source crop, figure assets, Markdown projection, and drawing provenance.
Retain useful page-level Qwen interpretation as evidence without making it
compete with the spatial-text pipeline.

Update mocked tests for minimal, multiple, zero, malformed, and regression
responses; Qwen coordinate conversion; no required order or transcription;
deterministic reading order; graphical-region exclusion; source-resolution
crop and asset provenance; TrOCR canonical text; zero-region behavior;
per-region failure; and the supplied five-bbox response. Do not use those
fixture coordinates as production page coordinates.

Do not redesign components, increase model output limits, lengthen the prompt,
change image preparation, change TrOCR or drawing localization, add semantic
correction, add a full-page fallback, or modify an ADR.

Validate with `just test`, `just lint`, `just check`, `sextant check`,
`sextant audit`, and `git diff --check`. Report implementation validation
separately from human real-model validation. The human integration command is
`just run samples/Grafting-101-page-004.png --markdown`.

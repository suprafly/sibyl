# Sibyl: spatial text localization → source-resolution transcription

You are Cody. Implement the next step in Sibyl's transform pipeline: use Qwen to identify spatial text regions, then transcribe those regions from original-resolution image crops with the existing TrOCR boundary.

Preserve the working drawing pipeline unchanged. Qwen page understanding must return page-level textual content, spatial text regions, and drawings. Spatial text regions use records with `order`, `kind`, `bbox_2d` in Qwen3-VL's `0..1000` coordinate space, and optional Qwen `text`. Qwen localizes; TrOCR is canonical for spatial text. Do not ask Qwen for canonical region transcription.

For each legitimate spatial text region, map Qwen `0..1000` coordinates through normalized coordinates, the existing 1536×2048 prepared image, and the existing prepared-to-source mapping to the original-resolution source image. Apply deterministic text-appropriate padding if available, clamp, crop the original image, and send only that crop to the existing TrOCR boundary. Never synthesize a full-page text region. Preserve order, Qwen bbox, prepared/source bounds, coordinate space, Qwen evidence, TrOCR text/status/timing, canonical text, and disagreement evidence.

Successful TrOCR text is canonical. A failed attempt is explicit and must not destroy the transform; preserve Qwen evidence without silently using it as canonical. If there are zero spatial text regions, do not fabricate any and record zero regions and zero TrOCR attempts while retaining page-level Qwen interpretation as explicit fallback evidence. Assemble canonical page text from successful spatial-region TrOCR results in reading order when available; retain page-level Qwen text as provenance.

Drawing localization remains authoritative for figures. Do not modify its prompt, schema, coordinate semantics, mapping, padding, crop implementation, original-resolution assets, or Markdown figure projection. Do not classify graphical arrows, strokes, connectors, cuts, or isolated drawing symbols as text. Do not duplicate figure regions as text regions.

Extend benchmark timings to preparation, page transform, drawing localization, text localization, TrOCR, crop, and total transform using existing naming conventions; track spatial text regions, TrOCR attempts, successes, and failures. Preserve the existing artifact model and Markdown figure syntax, using canonical assembled text without disagreement metadata in ordinary Markdown.

Add mocked regression tests covering spatial text localization (including zero, multiple, order, Qwen coordinates, prepared/source mapping, and original-resolution crops), TrOCR legitimacy, no synthetic full-page crops, canonical precedence, preserved evidence, disagreement, explicit failure, zero attempts, drawing isolation and unchanged figure assets, Markdown canonical projection, and JSON provenance. Do not hard-code specimen strings. Do not run Ollama, Qwen, Qwen3-VL, TrOCR, `sibyl run` on a real specimen, or any model inference. Do not create or modify an ADR.

Validate with `just test`, `just lint`, `just check`, `sextant check`, `sextant audit`, and `git diff --check`. Completion must distinguish Cody validation from human real-model validation and provide the exact human command: `just run samples/Grafting-101-page-004.png --markdown`.

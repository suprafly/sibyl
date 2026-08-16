# Sibyl: persist TrOCR input crops for inspection

Add deterministic debug/provenance artifacts for the spatial-text to TrOCR pipeline. For every spatial text region sent to TrOCR, persist the exact crop representation passed to TrOCR under `<stem>.sibyl/assets/text-NN.png`, named deterministically by reading order. Keep text assets separate from `figure-NN.png` assets and do not add them to ordinary Markdown.

Record per text region in transform.json: text asset path, source bounds, prepared bounds, Qwen bbox, coordinate space, padding, crop dimensions, and useful dimensions of the representation handed to TrOCR. If TrOCR preprocessing is internal, preserve the source crop and document that distinction without changing preprocessing. Use existing provenance/domain models.

Do not change canonical text precedence, Qwen prompts or schema, spatial localization, drawing localization, coordinate semantics, mapping, padding, source mapping, figure crops, figure Markdown, or figure provenance. Do not add fallback, semantic correction, dictionaries, or an ADR. Zero spatial regions produce zero text assets. A TrOCR failure must not prevent crop persistence.

Add mocked tests for text crop creation, deterministic names, figure separation, JSON provenance, bounds and dimensions, zero regions, persistence on TrOCR failure, no diagnostic text in Markdown, and unchanged drawing assets. Do not run model inference. Validate with `just test`, `just lint`, `just check`, `sextant check`, `sextant audit`, and `git diff --check`. Human follow-up: `just run samples/Grafting-101-page-004.png --markdown`, then inspect sorted files under `samples/Grafting-101-page-004.sibyl/assets/`.

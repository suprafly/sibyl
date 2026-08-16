# Amendment: accept prepared-image pixel drawing boxes

The dedicated drawing-localization adapter must accept both normalized drawing
bounding boxes and bounding boxes expressed directly in the prepared VLM image
dimensions. Coordinate space must be classified deterministically, preserved
as provenance, validated against the prepared image, and interpreted before the
existing padding, prepared-to-source mapping, original-source crop, and
Markdown projection steps. Page-level transform, TrOCR behavior, prompts,
preparation dimensions, and architecture decisions remain unchanged.

Regression coverage must include the observed prepared-image fixture
`[330, 707, 887, 872]` for a 1536×2048 prepared image, compatibility `bbox`
records, invalid ranges, metadata, source mapping, padding, original-resolution
cropping, and Markdown asset generation. Cody must not run model inference.

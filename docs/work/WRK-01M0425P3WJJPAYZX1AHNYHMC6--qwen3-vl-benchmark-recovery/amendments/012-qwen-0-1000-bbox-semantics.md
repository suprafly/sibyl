# Amendment: restore Qwen 0–1000 bbox semantics

The dedicated Qwen drawing-localization adapter has one explicit coordinate
contract: `bbox_2d` values are Qwen3-VL coordinates in the 0–1000 domain. They
must be converted to prepared-image coordinates before the existing
prepared-to-source mapping, padding, and original-resolution crop. The former
prepared-pixel interpretation and ambiguous automatic classification are
removed.

Provenance must identify the raw model space as `qwen_0_1000`. Existing prompts,
schemas, response extraction, drawing behavior, projections, artifacts, image
preparation, and benchmark structure remain unchanged. Regression coverage
must use the observed `[330, 707, 887, 872]` fixture without specimen-specific
production logic. No ADR change or model inference is authorized.

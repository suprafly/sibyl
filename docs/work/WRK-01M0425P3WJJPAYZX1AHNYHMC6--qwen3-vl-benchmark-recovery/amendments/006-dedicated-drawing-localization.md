# Amendment: dedicated drawing localization

Drawing discovery is now owned by a dedicated semantic Qwen3-VL localization
pass. Page recovery remains responsible for page-level text and must not
enumerate exhaustive text coordinates. The localization pass returns complete
normalized drawing boxes, which are deterministically padded, mapped, and
cropped from the original source. Page recovery and drawing localization remain
independently observable and failure-tolerant; no classical-CV primary detector,
OCR, model inference during Cody validation, or ADR changes are authorized.

# Amendment: revert spatial-text expansion

The current Qwen recovery boundary must request and normalize page-level
`text[]` plus spatial `drawing[]` with normalized bounding boxes. It must not
ask Qwen3-VL 8B to enumerate spatial text regions or make recovery depend on
spatial text localization. Page-level text is preserved exactly, does not
trigger TrOCR, and is projected directly by the default and Markdown outputs.
Legitimate drawing boxes continue to map through prepared-image coordinates to
original-source crops and Markdown/JSON assets. TrOCR remains available for a
future explicitly supported spatial-text response, but current page-level
responses have zero spatial text regions and zero TrOCR attempts. Mocked tests
must cover page-level preservation, no TrOCR, drawing coordinate projection,
one or multiple drawings, Markdown assets, truncated and unsupported JSON,
and valid page interpretation in `message.thinking`. No model inference or
ADR changes are authorized.

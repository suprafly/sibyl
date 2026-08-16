# Sibyl: increase experimental text-localization output budget

Change only the experimental region-first transcription-reread text-localizer
generation budget from `num_predict=128` to `num_predict=512`. Keep
`think=false`, `stream=false`, `keep_alive=0`, page transcription at 256,
regional OCR at 256, and all other generation controls unchanged. Do not
modify canonical `sibyl run`, page transcription, drawing localization, page
preparation, regional OCR, prompts, schemas, architecture, or add TrOCR or
sampling experiments.

Update the request-control test to assert 512 and retain/add a deterministic
multi-region response fixture. Preserve strict four-value bbox validation,
malformed-array rejection, truncated-response diagnosis, and raw responses.
Do not repair truncated JSON, truncate oversized arrays, or subdivide large
text regions. Run just test, just lint, just check, Sextant check/audit, and
git diff --check without real inference. Human command:

```fish
SIBYL_PAGE_FOCUS=full uv run sibyl experiment transcription-reread samples/Grafting-101-page-004.png
```

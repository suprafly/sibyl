# Sibyl: BOOX native-stroke-assisted handwriting recognition

Add `sibyl experiment boox-recognition` as an isolated, non-training experiment
for verified Grafting 101 page-4 native strokes. Compare baseline, native-render,
native-exemplar, multi-exemplar, and leave-one-region-out conditions through the
existing Qwen recognition path. Support existing region and line targeting,
deterministic native rendering, exact stroke provenance, repeated reads, raw
response preservation, and evaluation-only human transcription metrics. Never
send human transcriptions to the model. Do not change canonical recognition,
train, fine-tune, or invoke inference during implementation. Add deterministic
mocked tests and documentation; validate with the repository test, lint, check,
Sextant, audit, and diff-check commands.

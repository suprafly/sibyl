# Visual-exemplar handwriting experiment

`handwriting-exemplars` tests whether the existing multimodal Qwen recognizer
can use actual handwriting images from the same writer at inference time. It
does not train a model, add LoRA, or modify canonical `sibyl run` or
convergence.

A confirmed reference manifest connects each visual crop to its human
transcription for provenance. The transcription is never included in the
recognition prompt: only the reference pixels are shown. This makes visual
references different from a dictionary, which would inject text answers.
References must be explicitly confirmed, are sorted deterministically, and
the target crop is rejected as a reference by both path and hash. Individual
glyph crops are not invented; word or line crops are appropriate.

Each run compares a target-only baseline with deterministic prefixes of the
selected references: one, three, and five when available. The request keeps
the existing baseline Qwen controls fixed. Candidate frequency and stability
are observations, not correctness. Optional separately confirmed target
ground truth enables exact match, edit distance, and token-overlap metrics.

Example manifest:

```yaml
references:
  - id: reference-01
    crop: path/to/crop.png
    source_bbox: {left: 10, top: 20, right: 100, bottom: 50}
    transcription: "human-confirmed text"
    confirmed: true
```

The prompt labels reference images separately from the target and tells Qwen
to transcribe only the target. Reference words and target ground truth are
never sent to the model.

The artifact is written to `.sibyl/experiments/handwriting-exemplars.json`,
with generated experiment images under the corresponding artifact directory.
This is an inference-time visual adaptation experiment, not handwriting
training or future LoRA implementation.

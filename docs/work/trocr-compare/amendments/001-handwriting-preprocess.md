# Amendment: controlled handwriting crop preprocessing experiment

Add an experimental `sibyl experiment handwriting-preprocess IMAGE` benchmark
over existing transcription-reread/trocr-compare source crops. It must reuse
source-coordinate crop provenance and RGB pixels, support generic region
selection, optional existing line selection with line precedence, and explicit
crop paths where repository conventions permit. It must not change canonical
`sibyl run`, convergence, model configuration, or invoke an adjudicator.

Generate deterministic original RGB, grayscale, 2x/3x RGB, 2x grayscale,
contrast-normalized grayscale, and contrast-normalized 2x grayscale variants
without changing aspect ratio or original pixels. Run existing Qwen and TrOCR
adapters with preserved request contracts, default five repeated reads, and
preserve every raw, invalid, and truncated response. Record crop/variant
hashes, dimensions, parameters, recognizer configuration, timings, parsed and
normalized readings, stability, distinct readings, cross-variant summaries,
candidate support, optional separate human review, and deterministic exact,
normalized, token-overlap, and edit-distance metrics. Never encode a specimen
answer such as `shoot` in runtime logic.

Persist an auditable JSON artifact and generated variant images. Add mocked
tests for preprocessing, provenance, recognizer calls/configuration, response
preservation, aggregation, comparison, review/evaluation, determinism, and
canonical isolation. Document the experiment and validate without real model
inference.

# BOOX recognition matched-rendering follow-up

Run a focused follow-up experiment to test whether the verified BOOX native
strokes help Qwen when rendered in a representation matched to the target
image. Keep the verified decoder, identity coordinates, raw provenance, and
canonical Sibyl recognition unchanged. Improve the experiment-only native
reference rendering and selection: match native-reference stroke appearance
to the target raster, use consistent presentation scale and padding, and test
smaller carefully selected non-target reference sets. Preserve exact stroke
IDs, point counts, rendering parameters, hashes, target/reference separation,
raw responses, thinking, truncation evidence, and evaluation-only status.
Add deterministic mocked tests and documentation. Do not train, fine-tune,
send human transcriptions to Qwen, or run model inference during
implementation. Validate with the repository test, lint, check, Sextant, audit,
and diff-check commands.

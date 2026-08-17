# Amendment: distinguish graphical material from unresolved text

Extend experimental document-level convergence generically so unresolved or
weakly supported evidence is classified as `text`, `figure`, or `unknown`
using existing evidence only. Existing drawing/figure metadata, association,
geometry, explicit graphical recognition observations, and coherent textual
observations may support classification; do not call another model, add an
image classifier, use semantic knowledge, or add specimen-specific rules.

When source material is already represented by the existing figure projection,
prefer that representation and suppress duplicate textual uncertainty. Genuine
uncertain handwriting/text must remain `[unclear]`; unknown material must retain
uncertainty and provenance. Extend convergence JSON with classification kind,
basis, emitted status, and figure representation while preserving underlying
recognition evidence. Emit the existing figure projection exactly once and keep
its deterministic order. Preserve observed capitalization when supported by
evidence, including the grafting heading; do not resolve `on ↓` to `and`.

Add deterministic classification, projection, provenance, genericity, and
regression tests; update `docs/convergence.md`; preserve regional convergence,
human review, no-invention, and canonical `sibyl run` behavior. Do not run
real model inference. Validate with repository test/lint/check/Sextant/audit
commands and `git diff --check`.

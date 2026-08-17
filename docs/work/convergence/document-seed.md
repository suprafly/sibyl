# Seed

Extend the experimental convergence system with a second deterministic document-level pass over
regional candidates, alternatives, evidence, provenance, source geometry, and optional human review.
Preserve the regional layer and model evidence. Assemble regions in spatial reading order, support
adjacent textual continuity, score observed alternatives transparently, protect against stable-but-
wrong recognizers, preserve unresolved uncertainty, preserve figures, and enforce that every emitted
lexical token is supported by recognition evidence or explicit human review. Do not use an LLM,
semantic oracle, dictionaries, external knowledge, specimen-specific rules, model inference, or
changes to canonical Sibyl behavior, page preparation, drawing localization, Qwen, or TrOCR. Update
tests and docs/convergence.md. Fix no unrelated behavior. Follow the complete user-provided task
prompt as the task amendment.

# Sibyl architecture

Sibyl is a Python CLI-first local tool. It separates source evidence, recovered
structure, interpretation, vocabulary evidence, and projection. The durable
semantic object is a recovered note/page structure, not Markdown. Projection
writers consume that structure and must retain provenance back to the
authoritative source image.

## Boundaries

- Source evidence owns original scans/photos and deterministic references to
  source regions.
- Recovered structure owns text, uncertain or uninterpreted regions, drawings,
  diagrams, ordering, relationships, and provenance.
- Interpretation may propose recovered meaning, but models are assistants and
  never authorities. Provider details stay behind a future boundary.
- Vocabulary/glossary owns human-reviewed recognition evidence. Extracted
  candidates remain candidates until a human approves them.
- Projection converts recovered structure to targets such as Markdown/Obsidian
  or JSON without changing the recovery model.
- CLI/application orchestration owns command parsing and future use-case
  composition; it does not become the owner of recovery semantics.

Source authority and explicit uncertainty are invariants. A low-confidence
region may remain an image crop rather than becoming an unjustified
transcription. Numeric confidence thresholds are deliberately unspecified.

## Intended empirical slices

1. Glossary bootstrap: deterministic candidate extraction and filtering from a
   textual corpus, optional local classification, then reviewed and
   human-approved vocabulary.
2. Single-page recovery: one genuinely difficult handwritten page, including
   drawings and uncertain handwriting, recovered into structure and emitted to
   one initial projection.

Neither slice is implemented by the foundation CLI. Batch ingestion,
databases, vector stores, hosted APIs, and provider-specific inference are
outside the current repository shape.

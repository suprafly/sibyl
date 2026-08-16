---
type: adr
id: ADR-0002
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0001, ADR-0003, ADR-0004, ADR-0005, ADR-0006]
supersedes:
superseded_by:
---

# ADR-0002 - Text transformation and figure extraction are separate responsibilities

## Context

The source page contains both handwritten text and visual figures. Experiments
showed that asking one interpretation pass to transcribe handwriting while also
describing or localizing diagrams causes those responsibilities to interfere.

## Decision

Page-level textual transformation and figure localization/extraction are
separate transformation responsibilities over the same source image.

```text
                    SOURCE IMAGE
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       PAGE TRANSFORMATION   FIGURE LOCALIZATION
              │                     │
              ▼                     ▼
          page text             figure geometry
                                    │
                                    ▼
                              source crop
```

The page transformation must not be responsible for reconstructing figure
pixels. The figure localization pass must not become the canonical page
transcription mechanism.

## Consequences

- Page text and figure extraction remain independently testable.
- Drawing localization can evolve without changing page transcription.
- Page transcription can evolve without changing figure geometry.
- The same source image may be processed by multiple independent
  transformations.

## Options considered

### Combined page interpretation

Rejected because it gives one pass competing transcription and figure
responsibilities.

### Separate responsibilities

Accepted because each operation has a narrow, independently observable
contract.

## Notes

Anything useful but non-normative.

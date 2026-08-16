---
type: adr
id: ADR-0001
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0002, ADR-0003, ADR-0004, ADR-0005, ADR-0006]
supersedes:
superseded_by:
---

# ADR-0001 - Sibyl is a visual transformation system

## Context

Sibyl transforms handwritten pages and other visual documents that contain
multiple independent information classes: text, figures, annotations, spatial
relationships, and preserved visual evidence. Flattening that source into OCR
text loses information and conflates interpretation with evidence.

## Decision

Sibyl is a transformation system for visual documents, not an OCR system.

The canonical operation is:

```text
sibyl run IMAGE
```

A source visual artifact is transformed into a structured representation. The
transformation may contain textual information, graphical information, spatial
relationships, annotations, and preserved visual artifacts.

The transformation boundary is:

```text
source visual artifact
        ↓
structured transformation
```

## Consequences

- `sibyl run` is the canonical transformation operation.
- `transform.json` is the structured transformation representation.
- `transform.md` is a human-facing projection.
- Visual artifacts may remain as source-derived assets.
- OCR is an implementation technique for a particular transformation
  responsibility, not Sibyl's identity.

## Options considered

### OCR system

Rejected because it reduces visual documents to text and discards figures,
spatial relationships, and source evidence.

### Visual transformation system

Accepted because it preserves the document's multiple information classes in a
structured result.

## Notes

This decision is the foundation for the five related responsibilities recorded
in ADR-0002 through ADR-0006.

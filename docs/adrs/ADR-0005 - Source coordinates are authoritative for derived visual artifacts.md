---
type: adr
id: ADR-0005
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0001, ADR-0002, ADR-0003, ADR-0004, ADR-0006]
supersedes:
superseded_by:
---

# ADR-0005 - Source coordinates are authoritative for derived visual artifacts

## Context

Models operate on prepared or resized images, while source documents may have
substantially higher resolution. Persisting a crop from the prepared image
loses information and makes coordinate semantics ambiguous.

## Decision

Model-space geometry is an intermediate representation. Persisted visual
artifacts are derived using source-image coordinates and source-image pixels.

The coordinate flow is:

```text
model coordinates
      ↓
normalized coordinates
      ↓
prepared-image coordinates
      ↓
source-image coordinates
      ↓
source pixels
```

## Consequences

- Coordinate spaces must be explicit.
- Conversion must be deterministic.
- Source bounds must be recoverable.
- Persisted figure assets come from the original source image.
- Changes to model preparation do not inherently change source artifact
  resolution.

## Options considered

### Prepared-image artifacts

Rejected because prepared pixels may be lower resolution and are not the source
authority.

### Source-image artifacts

Accepted because deterministic mapping preserves source resolution and evidence.

## Notes

Anything useful but non-normative.

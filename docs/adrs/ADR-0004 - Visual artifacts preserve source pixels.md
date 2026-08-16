---
type: adr
id: ADR-0004
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0001, ADR-0002, ADR-0003, ADR-0005, ADR-0006]
supersedes:
superseded_by:
---

# ADR-0004 - Visual artifacts preserve source pixels

## Context

Model-generated descriptions of figures necessarily discard visual information.
The source figure can instead be preserved directly after its geometry has been
localized.

## Decision

When visual content can be preserved directly from the source, Sibyl prefers
source-derived visual artifacts over model-generated descriptions of that
content.

For figures:

```text
source figure
    ↓
localized geometry
    ↓
source-resolution crop
    ↓
figure asset
```

rather than:

```text
source figure
    ↓
model description
```

The figure asset is evidence, not an interpretation.

## Consequences

- Figure assets are derived from the source image.
- Model output identifies geometry rather than replacing the figure.
- Original-resolution source pixels are preferred for persisted visual
  artifacts.
- Markdown can embed the preserved figure directly.

## Options considered

### Model-generated figure description

Rejected as the primary persisted artifact because it loses source visual
information.

### Source-derived figure asset

Accepted because it retains the evidence while allowing interpretation to remain
separate.

## Notes

Anything useful but non-normative.

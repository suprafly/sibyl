---
type: adr
id: ADR-0006
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0001, ADR-0002, ADR-0003, ADR-0004, ADR-0005]
supersedes:
superseded_by:
---

# ADR-0006 - Transformation preserves interpretation and uncertainty

## Context

Model interpretation can be semantically plausible while being unfaithful to the
source. For example, the specimen output interpreted `Sap grafting` where the
visible wording was `Splice grafting`.

## Decision

Sibyl must distinguish source-derived content from model interpretation. It must
not silently replace uncertain model output with semantically inferred or
"corrected" content. Where interpretation is uncertain, that uncertainty should
remain representable in the transformation and provenance model.

In particular, Sibyl must not silently perform transformations such as:

```text
Sap → Splice
```

merely because `Splice` is semantically more plausible in context.

## Consequences

- Model interpretation remains distinguishable from source evidence.
- Semantic plausibility must not silently override observed content.
- Future confidence and disagreement mechanisms can build on this boundary.
- Human review can determine whether an uncertain interpretation should be
  corrected.

## Options considered

### Silent semantic correction

Rejected because it makes a plausible interpretation appear to be source fact.

### Explicit interpretation and uncertainty

Accepted because it preserves faithfulness and leaves correction to human
review.

## Notes

Anything useful but non-normative.

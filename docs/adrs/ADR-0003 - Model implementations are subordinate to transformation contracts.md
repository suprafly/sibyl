---
type: adr
id: ADR-0003
date: 2026-08-15
project: sibyl
status: accepted
tags: []
related: [ADR-0001, ADR-0002, ADR-0004, ADR-0005, ADR-0006]
supersedes:
superseded_by:
---

# ADR-0003 - Model implementations are subordinate to transformation contracts

## Context

Sibyl currently uses models for page transformation and figure localization,
but a model can be useful for one responsibility and poor for another. A
model-specific implementation must not define the durable architecture.

## Decision

Sibyl owns transformation responsibilities and contracts. Individual models are
implementation mechanisms underneath those contracts.

For example:

```text
page transformation
    → currently implemented with Qwen

figure localization
    → currently implemented with Qwen
```

does not mean:

```text
Qwen = architecture
```

The model can be replaced without changing the transformation responsibility or
its external contract.

## Consequences

- Model choice remains replaceable.
- Tests should target transformation contracts where practical.
- Model-specific behavior belongs at implementation boundaries.
- Prompts and model configuration are implementation details unless they
  establish a durable architectural constraint.

## Options considered

### Model-defined architecture

Rejected because it makes responsibilities and contracts depend on a replaceable
implementation.

### Contract-defined architecture

Accepted because it makes model replacement possible without changing Sibyl's
meaning.

## Notes

Anything useful but non-normative.

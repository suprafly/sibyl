# Amendment: Sibyl is a transform, not a transform operation

Sibyl's canonical operation is `sibyl run IMAGE`. The repository must describe
the system as transforming source imagery into a structured result and text,
Markdown, JSON, and figure-asset projections. The old `run` operation is
not a supported alias and must not appear in user-facing CLI help or current
usage documentation.

Rename transform terminology that belongs to Sibyl's transform domain across
CLI dispatch, domain models, projection writers, artifact filenames, runtime
labels, tests, fixtures, README, usage documentation, comments, and Work Item
documentation. The `<stem>.sibyl/` directory may remain, but transform-owned
artifacts should use transform-oriented names such as `transform.json` and
`transform.md`. Leave unrelated uses of transform unchanged when they do not
describe Sibyl's operation.

This is a terminology/domain-model cleanup. Preserve the working model,
drawing-localization, coordinate, crop, projection, provenance, timing, and
TrOCR boundaries. Do not add compatibility machinery, create or modify an ADR,
or run model inference.

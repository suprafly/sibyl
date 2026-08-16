# Amendment: fix Justfile transform forwarding

The `just run ...` recipe must invoke the canonical `sibyl run ...` operation.
It must forward the image and projection flags without requiring users to add
the subcommand themselves. This is a CLI/Justfile wiring correction only; the
transform pipeline, projections, model boundaries, and artifact behavior remain
unchanged.

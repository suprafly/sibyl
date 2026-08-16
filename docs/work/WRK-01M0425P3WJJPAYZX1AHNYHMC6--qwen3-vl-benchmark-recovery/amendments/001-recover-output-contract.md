# Amendment: recover output contract

The user-facing `sibyl recover IMAGE` contract is amended so the default
projection is human-readable recovered text, `--markdown` writes a durable
Markdown artifact with original-resolution figure assets, and `--json` emits
the existing complete structured recovery representation. `--markdown` and
`--json` are mutually exclusive. The internal recovery pipeline and Qwen
thinking-field handling remain unchanged; model boundaries stay mocked in
tests and no ADR is changed.

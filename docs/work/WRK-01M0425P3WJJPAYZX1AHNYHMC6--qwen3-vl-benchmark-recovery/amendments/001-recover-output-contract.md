# Amendment: run output contract

The user-facing `sibyl run IMAGE` contract is amended so the default
projection is human-readable transformed text, `--markdown` writes a durable
Markdown artifact with original-resolution figure assets, and `--json` emits
the existing complete structured transform representation. `--markdown` and
`--json` are mutually exclusive. The internal transform pipeline and Qwen
thinking-field handling remain unchanged; model boundaries stay mocked in
tests and no ADR is changed.

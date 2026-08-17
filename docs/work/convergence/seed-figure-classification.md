# Sibyl: distinguish unresolved text from graphical material during convergence

Implement the associated convergence amendment in
`amendments/003-figure-vs-unresolved-text.md`. The current Grafting-101
experimental convergence emits `[unclear]` after an existing
`![Figure 1](assets/figure-01.png)` projection. Add a generic evidence-based
classification layer (`text`, `figure`, `unknown`) so graphical material
already represented by a figure is retained through the figure projection and
not duplicated as textual uncertainty, while unresolved handwriting remains
`[unclear]`. Preserve evidence and explain the suppression in convergence.json.

Do not hard-code specimen IDs, page IDs, recognition strings, or invoke model
inference. Preserve regional convergence, alternatives, scoring, human review,
no-invention behavior, deterministic figure ordering, and canonical `sibyl run`.
Use observed evidence for capitalization; do not change `on ↓` to `and`.
Add focused deterministic tests and update `docs/convergence.md`. Run the
repository validation commands, including `just test`, `just lint`,
`just check`, `sextant check`, `sextant audit`, and `git diff --check` where
available.

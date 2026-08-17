# Sibyl: register as a BRFX app with an oracle-blue identity

Implement Sibyl's registration as a first-class BRFX application/project with
its own Design Bench identity. Register the existing repository through the
normal BRFX project metadata and machine-local registry mechanisms; do not
invent a new mechanism or add BRFX runtime identity to transformation outputs.

Inspect the existing BRFX registration/init/new flow, Design Bench project and
theme identity, project color registration, machine-local registry, `brfx
peep`, Aura project resolution and DMS publication, and an existing project
Aura already resolves. Use the same path and preserve the BRFX tooling -> Sibyl
project private boundary.

Create the dedicated, machine-readable identity `project: sibyl`, `theme:
sibyl`, with an oracle-blue/turquoise/electric-blue semantic accent near
`#18C8D8`, selected through the existing Design Bench vocabulary. Do not
scatter literal colors. The same identity must resolve coherent semantic
light and dark appearances with contrast validation; do not create arbitrary
separate identities.

Do not modify Aura with Sibyl-specific cases, maps, resolvers, detection, or
hard-coded names. Do not create a Sibyl Aura startup path, service, watcher, or
special reconciliation. Test whether generic Aura discovery resolves a newly
registered BRFX project. Fix only a clearly generic bug that benefits arbitrary
future projects. Classify the result as PASS if generic discovery works, or
ARCHITECTURAL GAP if registration succeeds but generic discovery does not.

Inspect Sibyl with the existing commands, including `brfx peep` and `brfx peep
sibyl` (or the actual equivalent), inspect Design Bench identity and Aura
status/resolution, and use generic reconciliation if needed. Add deterministic
tests for valid metadata/name, normal identity attachment, distinct semantic
accent, light/dark resolution, contrast, no scattered accent literals, BRFX
inspection, generic Aura resolution/reconciliation, absence of Sibyl-specific
Aura code/color mapping, and no BRFX provenance in transformation outputs.
Use fixtures/mocks if Aura cannot run deterministically and document the manual
acceptance test.

Document that Sibyl is a BRFX project with its own oracle-blue Design Bench
identity and is intentionally discovered by Aura's generic project-resolution
path, without claiming success until manually tested. Do not modify or create
ADRs. Run the repository's normal tests, lint, checks, Sextant check/audit,
BRFX validation, and `git diff --check`, using equivalents if recipes are
absent. Keep canonical Sibyl transformation behavior unchanged and clearly
separate Cody validation from human real-model validation; never run model
inference.

Completion must report registration, identity/accent, light/dark behavior,
BRFX inspection, Aura result and code changes, tests/validation, provenance,
and unchanged transformation behavior. Include exact manual commands and the
sentence: “Sibyl was / was not discovered by Aura through the existing generic
project-resolution mechanism without Sibyl-specific Aura integration.”

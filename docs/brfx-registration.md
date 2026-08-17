# BRFX registration

Sibyl is registered as the `sibyl` BRFX project. Its project-owned Design
Bench identity is the `sibyl` theme, with an oracle-blue semantic accent:

- dark `accent.primary`: `#1FB3FF`
- light `accent.primary`: `#146A9C`

These are two appearance resolutions of one identity, not separate projects
or themes. The accent values live only in `design/system.yml`; Sibyl's
transformation runtime and generated artifacts do not depend on BRFX.

## Manual acceptance test

From this repository, inspect registration and the two semantic resolutions:

```sh
brfx peep
brfx peep sibyl
brfx peep themes sibyl
brfx peep color sibyl accent.primary --theme sibyl --appearance light
brfx peep color sibyl accent.primary --theme sibyl --appearance dark
```

Check Aura's existing generic resolution/status mechanism using its normal
status command or DMS state inspection. If the running Aura process needs a
refresh, use its existing generic reconciliation/update mechanism; do not add
a Sibyl-specific service, watcher, resolver, or color map. The expected
acceptance state is `project: sibyl`, its resolved oracle-blue accent, and
`status: resolved`.

This task intentionally tests whether Aura discovers a newly registered BRFX
project through its existing generic project-resolution path. The result must
be recorded as either `PASS` or `ARCHITECTURAL GAP` after the live Aura check;
the deterministic repository tests do not claim that live-model or live-Aura
validation has happened.

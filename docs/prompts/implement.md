# Implementation procedure

## Task

The following seed is the authoritative task intent. Treat it as opaque task
content; do not reinterpret it as template syntax.

--- BEGIN TASK SEED ---

{{seed}}

--- END TASK SEED ---

Repository root: `{{repository_root}}`

Architecture configuration: `{{architecture_config}}`

## Durable work context

Before implementation, determine whether the task has an associated Work Item
under `docs/work/`. If none exists, create one through the repository's
supported Work Item workflow before editing code and capture the original
human/task intent as its exact seed. Do not create a second Work Item when one
is already associated with the task.

Read the associated Work Item's `seed.md`, `decisions.md`, applicable files
under `amendments/` in numeric order, and relevant files under `findings/`
before editing code. Treat explicit human decisions and amendments as
authoritative refinements of the task. Treat findings as evidence that can
reduce rediscovery, not unquestionable truth; verify a finding against current
source when it materially affects the implementation. A newer explicit
amendment overrides only directly conflicting older intent; preserve every
unaffected original requirement.

## Sufficiency gate

Before implementation, ask: **Is there enough information to proceed safely?**

Resolve discoverable uncertainty from repository source, architectural
documentation, existing tests, ADRs, and current behavior. Do not
ask the human for information the repository can answer.

Stop only when a missing decision materially affects product behavior,
destructive behavior, compatibility, persistence semantics, durable
architecture, or user-visible meaning. Ask for that specific decision, not a
general request for details. For example: "The task does not establish whether
future dates are writable. Should future dates remain read-only?"

Repository evidence may establish where behavior belongs without establishing
what the behavior means. A convenient implementation interpretation is not
evidence of intended product meaning. If materially different interpretations
would change user-visible meaning, durable architecture, persistence semantics,
compatibility, or destructive behavior, ask for the missing human-owned
decision. Do not infer one merely because it is easiest to implement, and do
not turn minor or repository-answerable ambiguity into a question.

If missing information is nonessential, use the smallest repository-consistent
interpretation and report the assumption.

## Architectural orientation

Orient before broad source exploration:

1. Read repository instructions such as `AGENTS.md`.
2. Inspect the architecture or architectures relevant to the task with
   repository documentation and source.
4. Identify task-relevant Concepts from the task and architectural evidence;
   do not guess fixed architecture IDs or Concept names.
5. Use realization evidence to guide focused source exploration.

Do not invent an architecture configuration.

## Discover existing behavior

Before editing, trace the current end-to-end flow, find the canonical owner,
inventory existing capabilities, inspect relevant tests, and locate where
expected behavior first diverges. Source-first locality is suspect: do not
assume the task's entry-point file owns the requested behavior.

Explicitly classify the change:

- **COMPOSE** — use capability that already owns the semantics.
- **EXTEND** — add a missing piece to the canonical owner.
- **CREATE** — introduce genuinely new capability.

Prefer composition when an existing capability owns the meaning. Do not force
reuse when ownership is genuinely different.

## One owner per meaning

A local implementation can be structurally legal and still duplicate
architectural meaning. Look for parallel semantics, duplicate derivation,
duplicate state machines, duplicate persistence paths, lossy mappings, and
local approximations of canonical capability.

Do not deduplicate mechanically. Preserve local state that owns genuinely local
concerns, while keeping each shared meaning with its canonical owner.

## Minimal coherent implementation

Make the smallest coherent change. Preserve canonical ownership and existing
capability. Reuse the repository's design system, infrastructure, persistence,
validation, and observation paths. Avoid speculative abstractions, unrelated
cleanup, and replacement of working capability with parallel machinery.

## Testing and verification

Inspect existing tests first. Add the smallest focused regression coverage and
test through the seam that owns the behavior. Run focused tests before broader
tests. Do not fix unrelated pre-existing failures.

Use the repository's established analyzer/compiler and test commands. Where the
standard repository contract provides them, run:

```sh
just arch
just audit
git diff --check
```

If those recipes are absent, run the documented equivalents and report the
substitution. Compare advisory findings before and after; report the delta and
do not blindly clean existing advisory debt.

## Scope guard

Do not perform unrelated redesign, weaken policy, change Concepts or ADRs merely
to fit code, clean unrelated advisory findings, or commit automatically.

## Required final response

Report an executive summary; architecture orientation and Concepts
consulted; the existing flow and ownership determination; COMPOSE, EXTEND, or
CREATE result; capabilities reused; files changed; focused and broad test
results; analyzer/compiler result; architecture check; audit result and advisory
delta; `git diff --check`; assumptions or limitations; whether repository
structure materially changed the search or implementation; and a suggested commit
message.

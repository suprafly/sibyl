# Implementation audit procedure

## Task intent

The seed may contain the original task, an implementation summary, known
concerns, or an audit focus. It has no required schema.

--- BEGIN TASK SEED ---

{{seed}}

--- END TASK SEED ---

Repository root: `{{repository_root}}`

Architecture configuration: `{{architecture_config}}`

## Durable work context

Determine whether the implementation has an associated Work Item under
`docs/work/`. If it does, read its `seed.md`, `decisions.md`, applicable ordered
`amendments/`, and relevant files under `findings/` before broad rediscovery.
The seed, decisions, and amendments are durable task intent. Findings are
evidence-backed conclusions, not unquestionable truth; verify materially
relevant findings against current repository state. Newer explicit amendments
override only direct conflicts, while unaffected original requirements remain.
Do not reconstruct intent solely from the final diff. Work Items are optional;
audit tasks without one use the supplied seed and repository evidence.

## Objective and sufficiency gate

Begin read-only. Determine whether the implementation is correct,
architecturally aligned, and unnecessarily inventive.

If task intent is insufficient to judge behavioral correctness, still evaluate
architecture where possible. Identify the exact missing behavioral or product
decision and do not invent a requirement. Resolve questions the repository can
answer from source, tests, ADRs, documentation, and current behavior.

## Architectural orientation

Before broad source exploration, read `AGENTS.md` and other repository
instructions. Discover relevant architectures and Concepts from the task, diff,
and architectural evidence. Follow realization evidence into source. Do not
assume fixed IDs, names, or config paths.

## Inspect the complete change

Audit the full diff rather than isolated files. Reconstruct the changed behavior
from entry point through state, persistence, side effects, and output. Inspect
the relevant tests and surrounding unchanged owners.

For each meaningful responsibility ask:

- Is this the correct owner?
- Was existing capability reused, and is new capability justified?
- Was a parallel system or duplicated semantic derivation introduced?
- Did a concrete implementation type leak upward?
- Was result or projection information lost?
- Does local state shadow canonical state?
- Was a persistence, validation, or observation path bypassed?
- Was the design system duplicated or approximated locally?

A structurally legal local implementation may still duplicate architectural
meaning. Preserve genuinely local concerns; do not demand mechanical
deduplication.

## Intent, symptom, and first-path survival

Decide whether the change solves the underlying problem or merely hides its
symptom. Answer: **Did the initial implementation path survive architectural
review, or does it require architectural redirection?**

Use exactly one verdict:

- **PASS**
- **PASS WITH MINOR CORRECTIONS**
- **ARCHITECTURAL REDIRECTION REQUIRED**

## Verification

Run focused validation and the repository's established analyzer/compiler and
test commands. Where available, run `just arch`, `just audit`, and
`git diff --check`; otherwise report the documented equivalents. Separate
existing advisory findings, the new advisory delta, and unrelated repository
debt.

## Audit boundary and report

Remain read-only. Do not opportunistically rewrite the implementation. Recommend
the minimum correction and its architectural owner when correction is needed.
This v1 audit prompt does not provide audit-and-fix mode.

The audit output is a transient execution result and must not be persisted in
full. At completion, identify any compact, evidence-backed conclusions durable
enough to append under an associated Work Item's `findings/`. Do not record
temporary hypotheses, reasoning, transcripts, or the full audit response. An
audit may establish zero, one, or multiple durable findings; record nothing
when no conclusion is worth preserving.

Report the verdict first, then findings ordered by severity with file evidence;
task sufficiency limitations; architecture orientation and Concepts
consulted; reconstructed flow; ownership and capability-reuse assessment;
focused and broad validation results; analyzer/compiler result; architecture
check; audit advisory baseline and delta; `git diff --check`; first-path
survival; minimum corrections; whether repository structure materially changed the audit;
and a suggested commit message when appropriate.

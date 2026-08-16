# Prompt amendment procedure

## Amendment input

The seed contains an existing prompt or task with sufficient context, plus an
amendment, correction, or new constraints. Prompt amendment before execution is
the primary use case.

--- BEGIN TASK SEED ---

{{seed}}

--- END TASK SEED ---

Repository root: `{{repository_root}}`

Architecture configuration: `{{architecture_config}}`

## Durable work context

Determine whether the task has an associated Work Item under `docs/work/`. If
it does, read its original `seed.md`, recorded `decisions.md`, ordered
`amendments/`, and relevant files under `findings/` as the durable baseline.
Findings are evidence to verify against current repository state when they
materially affect the amendment, not unquestionable truth. Newer explicit
amendments override only direct conflicts; all unaffected seed requirements and
decisions remain in force. Work Items are optional, and tasks without one
continue normally.

When the human supplies an explicit amendment for an associated Work Item, the
amendment belongs in the next ordered Markdown file under that Work Item's
`amendments/`. Do not collapse it into `decisions.md`, make tooling write it, or
persist assembled prompt output.

An amendment may invalidate or supersede an existing finding. Preserve the old
finding rather than rewriting it. Record a newer finding only when the updated,
evidence-backed conclusion is materially useful; never store temporary
hypotheses, reasoning, transcripts, or full agent output.

## Procedure

Treat the existing prompt or task as the authoritative baseline. Apply only the
amendment and preserve every unaffected requirement, including scope guards,
ownership rules, validation, and reporting requirements. Do not re-plan from
scratch or broaden scope.

Identify direct conflicts between older and newer instructions. Newer amendment
instructions override directly conflicting older instructions, but all
non-conflicting requirements remain intact. If explicit precedence resolves the
conflict, use the amendment as newer authority. If resolution requires a
product or architecture decision that the seed and repository evidence do not
establish, ask only for that specific decision. Do not ask the human for
information the repository can answer.

Return the amended prompt, or the amended implementation plan/instructions,
according to the seed's request. Preserve the baseline's structure where useful,
make the changed instruction unambiguous, and do not silently omit prior
constraints.

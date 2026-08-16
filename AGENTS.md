<!-- BEGIN MANAGED: engineering-work -->
## Engineering Work

Repository engineering methodology is available under `docs/prompts/`.
Substantive implementation tasks may use a Work Item under `docs/work/`. Create one with the original task as its seed before editing when none is already associated.

Before implementing or continuing a task with an associated Work Item:

1. Read its `seed.md`.
2. Read `decisions.md`.
3. Read applicable files under `amendments/` in numeric order.
4. Read relevant files under `findings/`; use them to reduce rediscovery, but verify them against current source when they materially affect the work.
5. Treat the seed, decisions, amendments, and current request as authoritative task context; findings are evidence, not authority.
6. Assemble and follow the appropriate repository prompt before editing code.
7. Do not infer human-owned product or durable architecture decisions for implementation convenience.
8. Record explicit human decisions and explicit task amendments in the Work Item as they occur. Preserve both the question and human answer in `decisions.md`; keep amendments distinct under `amendments/`.
9. When supported, record compact execution provenance; supply model identity only when it is explicitly available.
10. Record only durable, evidence-backed conclusions as findings.

Do not persist assembled prompts, chat transcripts, model reasoning, full audit responses, temporary hypotheses, temporary implementation plans, or architectural context dumps as Work Item state.
<!-- END MANAGED: engineering-work -->

# Sibyl: add controlled Qwen recognition-knob experiment

Add an evidence-gathering benchmark for `samples/Grafting-101-page-004.png` to determine whether
Qwen handwriting recognition improves through decoding controls, prompt formulation, and visual
context. Do not train, add LoRA, modify canonical `sibyl run`, convergence, dictionaries, hard-coded
`shoot`, or an LLM adjudicator. Do not run real model inference.

Before implementation, this seed must be a real file and must be assembled with `sextant prompt
assemble`; follow the assembled prompt. Reuse existing source crop and coordinate machinery. Support
existing region, existing line, and explicit crop targeting; line takes precedence; do not invent
specimen coordinates; default to the smallest existing crop for the ambiguous handwriting.

Separate these variables: (A) decoding controls temperature, top_p, seed, and num_predict, with
configurable temperature values 0.0, 0.2, 0.5, 0.8 and top_p values 1.0, 0.9, explicit deterministic
seed lists, and preserved existing defaults; (B) exact prompt variants consisting of the unchanged
existing regional transcription prompt, an isolated handwriting prompt, and an exact-word prompt;
record the exact prompt and do not hard-code the expected answer; (C) geometrically valid visual
contexts: tight, +5%, +10%, +20%, +30%, surrounding line, surrounding region, preserving source
coordinates and hashes.

Every experiment includes a documented baseline and records model, prompt, temperature, top_p, seed,
num_predict, think, stream, keep_alive, crop identity, and dimensions. Default to 5 runs (`--runs N`),
preserve explicit seeds, use deterministic seed lists, and preserve all raw successes, invalid,
truncated, and provider-failure responses with duration and available token counts.

For each result record `prompt_variant`, `context_variant`, decoding controls, and reading. Calculate
distinct readings, frequency, normalized readings, stability, first occurrence, and per-seed behavior;
call these candidates/support/frequency/stability, never correctness. Aggregate only observed
candidates across configurations and preserve optional external human review separately. Compute exact
normalized match, character edit distance, and token overlap only when explicit ground truth is
supplied; use no spellcheck, lexical database, web search, semantic model, or adjudicator.

Add `.sibyl/experiments/qwen-recognition-knobs.json`, CLI command
`sibyl experiment qwen-recognition-knobs IMAGE` with `--regions`, `--lines`, `--runs`, and useful
matrix-limiting controls, and documentation explaining purpose, variables, sampling, interpretation,
ground truth, and future training. Keep the default matrix manageable with staged prompt/context and
decoding phases; allow explicit expansion. Make request controls and artifact generation deterministic.

Add mocked tests covering targeting/precedence/provenance, context geometry and identity, prompts and
preservation, decoding sweeps and baseline controls, all response states, candidate analysis and
metrics, integrity/no-hard-coded-ground-truth/no-dictionary/no-adjudicator, unchanged canonical run,
and deterministic artifacts. Run `just test`, `just lint`, `just check`, `sextant check`, `sextant audit`,
and `git diff --check`. Completion must distinguish Cody validation from human real-model validation
and give the exact concise manual command for the ambiguous existing line plus artifact inspection.

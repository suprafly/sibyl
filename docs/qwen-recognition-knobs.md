# Qwen recognition-knob experiment

This experiment tests whether the existing Qwen model can produce better
handwriting recognition through prompting, visual context, and decoding
controls. It does not train the model, add LoRA, or change canonical
`sibyl run` or convergence.

It reuses persisted `transcription-reread` line crops and `trocr-compare`
region crops. `--lines` takes precedence over `--regions`; `--crop` targets an
explicit existing crop. Context crops are generated from the persisted source
bounds, with hashes and source coordinates retained. Invalid, truncated, and
provider-failure responses remain evidence alongside successful responses.

The three prompt variants are the unchanged regional prompt, an isolated
handwriting prompt, and an exact-word prompt. The default run is a manageable
prompt/context stage: three prompts, available context variants, and baseline
decoding controls, with five repeated runs per configuration. Use
`--decode-sweep` to test temperatures `0.0, 0.2, 0.5, 0.8`, `top_p` `1.0, 0.9`,
and deterministic seeds `0..4`, or provide narrower `--temperatures`,
`--top-p`, and `--seeds` values.

Candidate frequency, support, and stability describe observations. Frequency
is not correctness: the artifact never selects a most-frequent candidate as
truth. A human may provide a separate review file containing confirmed
`ground_truth`; only then are normalized exact match, character edit distance,
and token overlap calculated. No dictionary, spellcheck, web search, semantic
model, or adjudicator is used.

The evidence can show whether a reading appears under another prompt, context,
or decoding configuration, helping distinguish an elicitation failure from a
recognition limitation. Future handwriting training would be a separate effort;
this benchmark does not train the existing model.

Example:

```sh
uv run sibyl experiment qwen-recognition-knobs \
  samples/Grafting-101-page-004.png \
  --regions region-02 --lines region-02-line-04 --runs 5
```

Inspect the artifact with:

```sh
less .sibyl/experiments/qwen-recognition-knobs.json
```

# Convergence experiment

Sibyl’s experimental evidence path is:

```text
OBSERVE → PRESERVE → COMPARE → CONVERGE → PROJECT
```

`converge` consumes the preserved `.sibyl/experiments/trocr-compare.json` artifact. It does not call Qwen, TrOCR, another LLM, or any other recognizer. It compares repeated readings deterministically, normalizes presentation-only differences, clusters close lexical variants, extracts common token phrases, and emits a candidate Markdown projection plus JSON provenance. Model agreement is evidence, not truth: stable model output can still be wrong, and disagreement remains `[unclear]` unless the evidence is compatible or a human explicitly reviews it.

The JSON distinguishes model observations, recognizer stability, cross-model token overlap, common phrases, the candidate transcription, unresolved disagreements, source crop hashes, and human confirmation. A canonical page artifact, when discoverable beside the input source, is recorded as a page-level observation and may supply existing figure references; it is not silently promoted to ground truth.

## Run

```sh
./bin/sibyl experiment converge .sibyl/experiments/trocr-compare.json
```

This writes `.sibyl/experiments/converged.md` and `.sibyl/experiments/convergence.json`. Custom paths are available with `--output` and `--json-output`.

## Human review

Review is explicit and optional. Create a small YAML file:

```yaml
regions:
  region-02:
    text: "- transports mineral nutrients and water from root to shoot"
    confirmed: true
  region-05:
    text: "Splice grafting - what we will do now."
    confirmed: true
```

Then rerun deterministically with:

```sh
./bin/sibyl experiment converge .sibyl/experiments/trocr-compare.json \
  --review review.yaml
```

`confirmed: true` records an authoritative human confirmation for that candidate. `confirmed: false` records an explicit human suggestion but keeps the candidate unconfirmed. The review file is preserved by path in the JSON artifact; it is never inferred from model agreement.

The Markdown is experimental. It is not copied to `transform.md`, and the command does not modify canonical `sibyl run`, page transcription, localization, source crops, or figure assets.

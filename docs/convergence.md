# Convergence experiment

Sibyl’s experimental evidence path is:

```text
OBSERVE → PRESERVE → COMPARE → REGIONAL CONVERGENCE → DOCUMENT CONVERGENCE → PROJECT
```

`converge` consumes the preserved `.sibyl/experiments/trocr-compare.json` artifact. It does not call Qwen, TrOCR, another LLM, or any other recognizer. Regional convergence compares repeated readings deterministically, normalizes presentation-only differences, clusters close lexical variants, and extracts common token phrases. A second document pass then uses spatial order, observed alternatives, page-level observations when available, and local continuity to choose among already-supported candidates. Model agreement is evidence, not truth: stable model output can still be wrong, and disagreement remains `[unclear]` unless the evidence is compatible or a human explicitly reviews it.

Every emitted lexical token must have an evidence path to a regional recognition observation, an available page-level observation, or explicit human review. Document context can select among observed alternatives or extend a supported phrase with observed page-level material; it cannot invent vocabulary. A document is not merely a bag of independently recognized regions, so the JSON preserves both the regional candidate and the document-level decision and scoring basis.

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

## Unresolved text and graphical material

Document convergence classifies weakly supported material from preserved
evidence as `text`, `figure`, or `unknown`:

```text
uncertain text
    → [unclear]

graphical material already represented by figure
    → figure projection

unknown material
    → preserve uncertainty/provenance
```

Existing drawing-region metadata, figure association, source geometry, and
explicit graphical recognition evidence can associate a source crop with a
figure. When that association is strong enough, the figure projection remains
authoritative and the textual candidate is not emitted a second time. The
recognition observations remain in `convergence.json` with a classification,
its evidence basis, emission status, and figure representation so the
suppression is inspectable. Uncertain handwriting without positive graphical
evidence remains `[unclear]`; convergence does not infer text from a figure or
replace unresolved handwriting with a semantic guess.

# Transcription variance experiment

This investigation repeats the existing Qwen page-transcription request
against one prepared page image. It measures how much the model's parsed page
transcription changes between otherwise equivalent requests. It does not
measure handwriting quality, prove that repeated sampling improves results, or
choose a correct answer.

The harness prepares the source page once, hashes the exact lossless prepared
PNG representation, and reuses that representation for every page request.
It preserves each raw Ollama response, including malformed, truncated, and
failed responses. Drawing localization, figure extraction, and canonical
`transform.json`/`transform.md` generation are not part of this experiment.

## Run it

The default is five runs. The count can be set with
`SIBYL_TRANSCRIPTION_RUNS` or overridden with `--runs`:

```fish
SIBYL_PAGE_FOCUS=full SIBYL_TRANSCRIPTION_RUNS=5 uv run sibyl experiment transcription-variance samples/Grafting-101-page-004.png
```

The command accepts any page image. To write somewhere else, use
`--output path/to/result.json`.

By default the experimental result is written to:

```text
.sibyl/experiments/transcription-variance.json
```

The terminal output lists `Run 1` through `Run N`, shows successful
transcriptions, reports invalid or failed runs, and says whether successful
parsed strings are identical or different. The JSON retains the prepared image
dimensions and hash, request controls, duration, status, text, error, and raw
response for every run. Inspect the individual transcriptions with:

```sh
uv run python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path('.sibyl/experiments/transcription-variance.json').read_text())
for run in result['runs']:
    print(f"Run {run['run']} [{run['status']}]")
    print(run.get('text') or run.get('error') or '<no text>')
    print()
print('comparison:', result['comparison'])
print('failures:', result['failure_summary'])
PY
```

The experiment keeps sampling controls at the existing page-interpreter
values: `think=false`, `num_predict=256`, `stream=false`, and `keep_alive=0`.
Temperature, `top_p`, and seed remain unspecified Ollama/model defaults. It
uses the existing page prompt and schema without modifying them, so this is a
measurement of the current request behavior.

If successful outputs are identical, stochastic variance is probably not the
main explanation for the observed errors. If they differ, the result provides
evidence for a future candidate-comparison experiment. If requests fail or
truncate, interpret that failure rate separately from transcription
disagreement. This experiment performs no adjudication or automatic winner
selection.

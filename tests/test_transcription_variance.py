import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from sibyl.experiments.transcription_variance import (
    VarianceResult,
    format_variance_result,
    run_variance_experiment,
)
from sibyl.transform import PreparedVlmImage, prepare_page_image_with_metadata


class FakeInterpreter:
    model = "qwen3-vl:8b"

    def __init__(self, observer: Callable[[dict[str, Any]], None], responses: list[Any]) -> None:
        self.response_metadata: dict[str, Any] = {}
        self._observer = observer
        self._responses = responses
        self.calls: list[Image.Image] = []
        self.released = False

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.calls.append(image)
        response = self._responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        raw, interpretation = response
        self._observer(raw)
        self.response_metadata = {"raw_response": raw}
        return interpretation, 4.0

    def release(self) -> None:
        self.released = True


def _factory_for(
    responses: list[Any], created: list[FakeInterpreter]
) -> Callable[[Callable[[dict[str, Any]], None]], FakeInterpreter]:
    def factory(observer: Callable[[dict[str, Any]], None]) -> FakeInterpreter:
        interpreter = FakeInterpreter(observer, responses)
        created.append(interpreter)
        return interpreter

    return factory


def _valid(raw: dict[str, Any], text: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    return raw, {"page_interpretation": {"text": text}}


def _page(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(path)
    return path


def test_reuses_one_prepared_image_and_preserves_raw_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = _page(tmp_path)
    original_prepare = prepare_page_image_with_metadata
    preparations = 0

    def prepare(source: Image.Image) -> PreparedVlmImage:
        nonlocal preparations
        preparations += 1
        return original_prepare(source)

    monkeypatch.setattr(
        "sibyl.experiments.transcription_variance.prepare_page_image_with_metadata", prepare
    )
    created: list[FakeInterpreter] = []
    result = run_variance_experiment(
        image_path,
        runs=3,
        output_path=tmp_path / "result.json",
        interpreter_factory=_factory_for(
            [
                _valid({"message": {"content": "raw one"}}, ["one"]),
                _valid({"message": {"content": "raw two"}}, ["two"]),
                _valid({"message": {"content": "raw three"}}, ["one"]),
            ],
            created,
        ),
    )

    assert preparations == 1
    assert len(created) == 1
    assert len(created[0].calls) == 3
    assert created[0].calls[0] is created[0].calls[1] is created[0].calls[2]
    assert len({result.prepared_image_hash}) == 1
    assert [run.raw_response for run in result.runs] == [
        {"message": {"content": "raw one"}},
        {"message": {"content": "raw two"}},
        {"message": {"content": "raw three"}},
    ]
    assert result.comparison["transcriptions_different"] is True
    assert json.loads((tmp_path / "result.json").read_text())["runs_completed"] == 3


def test_reports_valid_invalid_truncated_and_failed_runs(tmp_path: Path) -> None:
    created: list[FakeInterpreter] = []
    result = run_variance_experiment(
        _page(tmp_path),
        runs=4,
        output_path=tmp_path / "result.json",
        interpreter_factory=_factory_for(
            [
                _valid({"message": {"content": "complete"}}, ["ok"]),
                (
                    {"message": {"content": '{"page_interpretation":'}},
                    {
                        "status": "failure",
                        "error": "no valid structured JSON",
                        "raw_response": {"truncated": True},
                    },
                ),
                (
                    {"message": {"thinking": "partial"}},
                    {
                        "status": "failure",
                        "error": "truncated thinking",
                        "raw_response": {"partial": True},
                    },
                ),
                RuntimeError("connection failed"),
            ],
            created,
        ),
    )

    assert [run.status for run in result.runs] == [
        "ok",
        "invalid_response",
        "invalid_response",
        "failed",
    ]
    assert result.runs[0].text == "ok"
    assert result.runs[1].text is None
    assert result.runs[2].raw_response == {"message": {"thinking": "partial"}}
    assert result.runs[3].error == "connection failed"
    assert result.failure_summary == {"ok": 1, "invalid_response": 2, "failed": 1}


def test_identical_reporting_excludes_failures(tmp_path: Path) -> None:
    result = run_variance_experiment(
        _page(tmp_path),
        runs=2,
        output_path=tmp_path / "result.json",
        interpreter_factory=_factory_for(
            [
                _valid({"message": {"content": "a"}}, ["same"]),
                _valid({"message": {"content": "b"}}, ["same"]),
            ],
            [],
        ),
    )

    assert result.comparison["transcriptions_identical"] is True
    assert "Successful parsed transcriptions are identical." in format_variance_result(result)


def test_run_count_and_page_focus_are_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIBYL_TRANSCRIPTION_RUNS", "2")
    monkeypatch.setenv("SIBYL_PAGE_FOCUS", "content")
    created: list[FakeInterpreter] = []
    result = run_variance_experiment(
        _page(tmp_path),
        output_path=tmp_path / "result.json",
        interpreter_factory=_factory_for(
            [
                _valid({"message": {"content": "a"}}, ["a"]),
                _valid({"message": {"content": "b"}}, ["b"]),
            ],
            created,
        ),
    )

    assert result.runs_requested == 2
    assert result.page_focus == "content"
    assert result.request_controls == {
        "model": "qwen3-vl:8b",
        "think": False,
        "num_predict": 256,
        "stream": False,
        "keep_alive": 0,
        "temperature": "unspecified (Ollama/model default)",
        "top_p": "unspecified (Ollama/model default)",
        "seed": "unspecified (Ollama/model default)",
        "prompt": "existing OllamaPageInterpreter page-transcription prompt",
        "schema": "existing OllamaPageInterpreter page-transcription schema",
    }


def test_no_drawing_localizer_is_needed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("drawing localization must not run")

    monkeypatch.setattr("sibyl.transform.OllamaDrawingLocalizer", fail_if_called)
    result = run_variance_experiment(
        _page(tmp_path),
        runs=1,
        output_path=tmp_path / "result.json",
        interpreter_factory=_factory_for(
            [_valid({"message": {"content": "raw"}}, ["text"])], []
        ),
    )
    assert isinstance(result, VarianceResult)

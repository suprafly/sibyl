import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from sibyl.experiments.qwen_recognition_knobs import (
    EXACT_WORD_PROMPT,
    ISOLATED_PROMPT,
    aggregate_candidates,
    run_qwen_recognition_knobs,
)
from sibyl.experiments.transcription_reread import REGIONAL_PROMPT


class FakeReader:
    model = "qwen-test"
    instances: ClassVar[list["FakeReader"]] = []
    observations: ClassVar[list[tuple[str, dict[str, Any], tuple[int, int]]]] = []
    number = 0

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer
        self.instances.append(self)

    def read(
        self, image: Image.Image, prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        type(self).number += 1
        self.observations.append((prompt, controls.copy(), image.size))
        if type(self).number == 2:
            raw: dict[str, Any] = {"message": {"content": "not-json"}}
            self.observer(raw)
            return {"status": "invalid_response", "raw_response": raw, "error": "missing text"}, 2.0
        if type(self).number == 3:
            raw = {"done_reason": "length", "message": {"content": "{}"}}
            self.observer(raw)
            return {"status": "truncated_response", "raw_response": raw}, 3.0
        if type(self).number == 4:
            raise RuntimeError("provider down")
        raw = {"message": {"content": '{"text": "alpha"}'}, "eval_count": 4}
        self.observer(raw)
        return {"status": "ok", "text": "alpha", "raw_response": raw}, 1.0

    def release(self) -> None:
        return None


def _crop_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "page.png"
    crop = tmp_path / "crop.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(source)
    Image.new("RGB", (10, 5), (4, 5, 6)).save(crop)
    return source, crop


def test_prompts_contexts_and_raw_response_states_are_recorded(tmp_path: Path) -> None:
    source, crop = _crop_fixture(tmp_path)
    FakeReader.instances = []
    FakeReader.observations = []
    FakeReader.number = 0
    artifact = run_qwen_recognition_knobs(
        source,
        crop_path=crop,
        runs=4,
        prompt_variants=("regional", "isolated", "exact-word"),
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeReader,
    )
    assert artifact["prompts"] == {
        "regional": REGIONAL_PROMPT,
        "isolated": ISOLATED_PROMPT,
        "exact-word": EXACT_WORD_PROMPT,
    }
    assert len(artifact["results"]) == 3
    assert artifact["results"][0]["analysis"]["runs"][1]["status"] == "invalid_response"
    assert artifact["results"][0]["analysis"]["runs"][2]["status"] == "truncated_response"
    assert artifact["results"][0]["analysis"]["runs"][3]["status"] == "provider_failure"
    assert artifact["results"][0]["analysis"]["runs"][0]["raw_response"]
    assert all(item[0] in artifact["prompts"].values() for item in FakeReader.observations)
    assert artifact["results"][0]["decoding_controls"] == {
        "temperature": None,
        "top_p": None,
        "seed": None,
        "num_predict": 256,
        "think": False,
        "stream": False,
        "keep_alive": 0,
    }


def test_context_padding_preserves_source_coordinates_and_hash(tmp_path: Path) -> None:
    source, crop = _crop_fixture(tmp_path)
    compare = tmp_path / "compare.json"
    compare.write_text(
        json.dumps(
            {
                "experiment": "trocr_compare",
                "source": str(source),
                "regions": [
                    {
                        "region_id": "region-1",
                        "crop": {
                            "path": str(crop),
                            "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                            "source_bbox": {"left": 10, "top": 5, "right": 20, "bottom": 10},
                            "source_coordinate_space": "source",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = run_qwen_recognition_knobs(
        source,
        regions="region-1",
        contexts=("tight", "padding-20"),
        runs=1,
        prompt_variants=("isolated",),
        compare_artifact=compare,
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeReader,
    )
    contexts = artifact["targets"][0]["contexts"]
    assert [item["variant"] for item in contexts] == ["tight", "padding-20"]
    assert contexts[0]["source_bbox"] == {"left": 10, "top": 5, "right": 20, "bottom": 10}
    assert contexts[1]["dimensions"] == {"width": 14, "height": 7}
    assert (
        contexts[0]["sha256"] == hashlib.sha256(Path(contexts[0]["path"]).read_bytes()).hexdigest()
    )


def test_candidate_aggregation_reports_observed_support_only() -> None:
    results = [
        {
            "prompt_variant": "isolated",
            "context_variant": "tight",
            "decoding_controls": {"seed": 1},
            "analysis": {
                "candidates": [{"candidate": "alpha", "normalized": "alpha", "frequency": 2}]
            },
        },
        {
            "prompt_variant": "regional",
            "context_variant": "padding-10",
            "decoding_controls": {"seed": 2},
            "analysis": {
                "candidates": [{"candidate": "beta", "normalized": "beta", "frequency": 1}]
            },
        },
    ]
    summary = aggregate_candidates(results)
    assert summary[0]["candidate"] == "alpha"
    assert summary[0]["frequency"] == 2
    assert summary[0]["support"][0]["prompt_variant"] == "isolated"
    assert {item["candidate"] for item in summary} == {"alpha", "beta"}


def test_decoding_sweep_preserves_baseline_and_explicit_controls(tmp_path: Path) -> None:
    source, crop = _crop_fixture(tmp_path)
    FakeReader.number = 0
    artifact = run_qwen_recognition_knobs(
        source,
        crop_path=crop,
        runs=1,
        prompt_variants=("isolated",),
        temperatures=(0.0, 0.2),
        top_ps=(1.0, 0.9),
        seeds=(None, 7),
        num_predict=99,
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeReader,
    )
    controls = [result["decoding_controls"] for result in artifact["results"]]
    assert len(controls) == 1 + (2 * 2 * 2 - 1)
    assert controls[0]["temperature"] is None
    assert controls[0]["num_predict"] == 99
    assert {item["seed"] for item in controls} == {None, 7}
    assert {item["temperature"] for item in controls} == {None, 0.0, 0.2}
    assert {item["top_p"] for item in controls} == {None, 1.0, 0.9}

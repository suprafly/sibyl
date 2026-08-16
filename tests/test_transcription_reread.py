from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

import sibyl.experiments.transcription_variance as variance_module
from sibyl.experiments.transcription_reread import (
    LOCALIZATION_SCHEMA,
    TARGETED_PROMPT,
    TARGETED_SCHEMA,
    format_reread_result,
    run_reread_experiment,
)
from sibyl.transform import PreparedVlmImage, prepare_page_image_with_metadata


class FakePageInterpreter:
    model = "qwen3-vl:8b"

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.response_metadata: dict[str, Any] = {}
        self.observer = observer
        self.images: list[Image.Image] = []

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.images.append(image)
        raw = {"message": {"content": "page raw"}}
        self.observer(raw)
        lines = [["stable", "scion"], ["stable", "scler"], ["stable", "stem"]][
            len(self.images) - 1
        ]
        return {"page_interpretation": {"text": lines}}, 1.0

    def release(self) -> None:
        return None


class FakeLocalizer:
    model = "qwen3-vl:8b"

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer
        self.images: list[Image.Image] = []

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.images.append(image)
        self.observer({"message": {"content": "localization raw"}})
        return {
            "text_regions": [
                {"order": 0, "bbox_2d": [100, 100, 300, 300]},
                {"order": 1, "bbox_2d": [100, 400, 300, 600]},
            ]
        }, 2.0

    def release(self) -> None:
        return None


class FakeRereader:
    model = "qwen3-vl:8b"
    instances: ClassVar[list["FakeRereader"]] = []

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer
        self.calls: list[Image.Image] = []
        self.__class__.instances.append(self)

    def reread(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.calls.append(image)
        raw = {"message": {"content": "reread raw"}}
        self.observer(raw)
        return {"text": f"reread-{len(self.calls)}"}, 3.0

    def release(self) -> None:
        return None


def _page(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), (12, 34, 56)).save(path)
    return path


def _page_factory(created: list[FakePageInterpreter]) -> Callable[..., FakePageInterpreter]:
    def factory(observer: Callable[[dict[str, Any]], None]) -> FakePageInterpreter:
        interpreter = FakePageInterpreter(observer)
        created.append(interpreter)
        return interpreter

    return factory


def test_disagreement_localizes_once_crops_source_and_rereads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_drawing_localizer(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("drawing localization must not run")

    monkeypatch.setattr("sibyl.transform.OllamaDrawingLocalizer", fail_drawing_localizer)
    original_prepare = prepare_page_image_with_metadata
    preparations = 0

    def prepare(source: Image.Image) -> PreparedVlmImage:
        nonlocal preparations
        preparations += 1
        return original_prepare(source)

    monkeypatch.setattr(variance_module, "prepare_page_image_with_metadata", prepare)
    page_created: list[FakePageInterpreter] = []
    localizer_created: list[FakeLocalizer] = []
    FakeRereader.instances = []

    def localizer_factory(observer: Callable[[dict[str, Any]], None]) -> FakeLocalizer:
        localizer = FakeLocalizer(observer)
        localizer_created.append(localizer)
        return localizer

    result = run_reread_experiment(
        _page(tmp_path),
        runs=3,
        rereads=2,
        output_path=tmp_path / "transcription-reread.json",
        localizer_factory=localizer_factory,
        rereader_factory=FakeRereader,
        interpreter_factory=_page_factory(page_created),
    )

    assert preparations == 1
    assert len(page_created) == 1
    assert page_created[0].images[0] is page_created[0].images[1] is page_created[0].images[2]
    assert len(localizer_created) == 1
    assert len(localizer_created[0].images) == 1
    assert result["prepared_image_hash"]
    assert len(result["disagreements"]) == 1
    disagreement = result["disagreements"][0]
    assert disagreement["page_candidates"] == ["scion", "scler", "stem"]
    assert disagreement["bbox"]["space"] == "qwen_0_1000"
    assert disagreement["crop"]["coordinate_space"] == "source"
    assert disagreement["crop"]["width"] == 22
    assert disagreement["crop"]["height"] == 22
    assert Path(disagreement["crop"]["path"]).exists()
    assert len(FakeRereader.instances) == 1
    assert len(FakeRereader.instances[0].calls) == 2
    assert disagreement["rereads"][0]["raw_response"] == {
        "message": {"content": "reread raw"}
    }
    assert result["localization"]["status"] == "success"
    assert "scler" not in TARGETED_PROMPT


def test_stable_page_has_no_disagreement_and_no_localization(tmp_path: Path) -> None:
    class StablePage(FakePageInterpreter):
        def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            self.images.append(image)
            return {"page_interpretation": {"text": ["stable"]}}, 1.0

    localizer_called = False

    def localizer_factory(observer: Callable[[dict[str, Any]], None]) -> FakeLocalizer:
        nonlocal localizer_called
        localizer_called = True
        return FakeLocalizer(observer)

    def page_factory(observer: Callable[[dict[str, Any]], None]) -> StablePage:
        return StablePage(observer)

    result = run_reread_experiment(
        _page(tmp_path),
        runs=3,
        output_path=tmp_path / "result.json",
        localizer_factory=localizer_factory,
        rereader_factory=FakeRereader,
        interpreter_factory=page_factory,
    )

    assert result["disagreements"] == []
    assert result["localization"] == {"status": "not_needed"}
    assert localizer_called is False


def test_unavailable_localization_preserves_candidates_without_crop(tmp_path: Path) -> None:
    class InvalidLocalizer:
        model = "qwen3-vl:8b"

        def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            return {
                "status": "failure",
                "error": "malformed localization",
                "raw_response": {"partial": True},
            }, 1.0

        def release(self) -> None:
            return None

    result = run_reread_experiment(
        _page(tmp_path),
        runs=3,
        output_path=tmp_path / "result.json",
        localizer_factory=lambda observer: InvalidLocalizer(),
        rereader_factory=FakeRereader,
        interpreter_factory=_page_factory([]),
    )

    assert result["disagreements"][0]["page_candidates"]
    assert "crop" not in result["disagreements"][0]
    assert result["localization"]["status"] == "unavailable"
    assert "targeted localization unavailable" in result["localization"]["message"]
    assert "candidate disagreement detected" in format_reread_result(result)


def test_targeted_schema_is_minimal_and_has_no_bbox_or_candidate_fields() -> None:
    assert TARGETED_SCHEMA == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    assert "bbox" not in TARGETED_SCHEMA["properties"]
    assert "candidate" not in TARGETED_PROMPT.lower()
    assert "text_regions" in LOCALIZATION_SCHEMA["properties"]

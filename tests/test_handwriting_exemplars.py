import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from sibyl.experiments.handwriting_exemplars import run_handwriting_exemplars


class FakeExemplarReader:
    model = "qwen-test"
    calls: ClassVar[list[tuple[str, list[tuple[int, int]]]]] = []
    run_number: ClassVar[int] = 0

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer

    def read(
        self, images: list[Image.Image], prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        type(self).run_number += 1
        self.calls.append((prompt, [image.size for image in images]))
        if type(self).run_number == 2:
            raw: dict[str, Any] = {"message": {"content": "not-json"}}
            self.observer(raw)
            return {"status": "invalid_response", "raw_response": raw, "error": "missing text"}, 2.0
        if type(self).run_number == 3:
            raw = {"done_reason": "length", "message": {"content": "{}"}}
            self.observer(raw)
            return {"status": "truncated_response", "raw_response": raw}, 3.0
        raw = {"message": {"content": '{"text": "alpha"}'}}
        self.observer(raw)
        return {"status": "ok", "text": "alpha", "raw_response": raw}, 1.0

    def release(self) -> None:
        return None


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "page.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(source)
    Image.new("RGB", (10, 5), (4, 5, 6)).save(target)
    references = []
    for identifier, color in (
        ("reference-b", (10, 11, 12)),
        ("reference-a", (13, 14, 15)),
        ("reference-c", (16, 17, 18)),
    ):
        crop = tmp_path / f"{identifier}.png"
        Image.new("RGB", (8, 4), color).save(crop)
        references.append(
            {
                "id": identifier,
                "crop": crop.name,
                "source_bbox": {"left": 1, "top": 2, "right": 9, "bottom": 6},
                "transcription": identifier,
                "confirmed": True,
            }
        )
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({"references": references}), encoding="utf-8")
    return source, target, manifest


def test_target_and_reference_sets_are_provenanced_and_ordered(tmp_path: Path) -> None:
    source, target, manifest = _fixture(tmp_path)
    FakeExemplarReader.calls = []
    FakeExemplarReader.run_number = 0
    artifact = run_handwriting_exemplars(
        source,
        target_crop=target,
        reference_manifest=manifest,
        references="reference-b,reference-a,reference-c",
        runs=1,
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeExemplarReader,
    )
    assert [item["set_id"] for item in artifact["reference_sets"]] == [
        "baseline",
        "references-01",
        "references-03",
    ]
    assert artifact["reference_sets"][1]["reference_ids"] == ["reference-a"]
    assert artifact["reference_sets"][2]["reference_ids"] == [
        "reference-a",
        "reference-b",
        "reference-c",
    ]
    assert artifact["target"]["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert artifact["references"][0]["transcription"] == "reference-a"
    assert [len(images) for _prompt, images in FakeExemplarReader.calls] == [1, 2, 4]
    assert all(images[-1] == (10, 5) for _prompt, images in FakeExemplarReader.calls)


def test_reference_transcriptions_never_enter_exemplar_prompt(tmp_path: Path) -> None:
    source, target, manifest = _fixture(tmp_path)
    FakeExemplarReader.calls = []
    FakeExemplarReader.run_number = 0
    run_handwriting_exemplars(
        source,
        target_crop=target,
        reference_manifest=manifest,
        references="reference-a",
        reference_set="reference-a",
        runs=1,
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeExemplarReader,
    )
    exemplar_prompt = FakeExemplarReader.calls[-1][0]
    assert "REFERENCE IMAGES" in exemplar_prompt
    assert "TARGET IMAGE" in exemplar_prompt
    assert "reference-a" not in exemplar_prompt


def test_invalid_and_truncated_responses_are_preserved(tmp_path: Path) -> None:
    source, target, _manifest = _fixture(tmp_path)
    FakeExemplarReader.calls = []
    FakeExemplarReader.run_number = 0
    artifact = run_handwriting_exemplars(
        source,
        target_crop=target,
        runs=3,
        output_path=tmp_path / "artifact.json",
        reader_factory=FakeExemplarReader,
    )
    analysis = artifact["results"][0]["analysis"]
    assert [item["status"] for item in analysis["runs"]] == [
        "ok",
        "invalid_response",
        "truncated_response",
    ]
    assert analysis["invalid_count"] == 1
    assert analysis["truncated_count"] == 1
    assert analysis["candidates"] == [
        {"candidate": "alpha", "normalized": "alpha", "frequency": 1, "stability": 1 / 3}
    ]


def test_target_reference_identity_and_confirmation_are_required(tmp_path: Path) -> None:
    source, target, manifest = _fixture(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["references"][0]["crop"] = target.name
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="target crop"):
        run_handwriting_exemplars(
            source,
            target_crop=target,
            reference_manifest=manifest,
            references="reference-b",
            output_path=tmp_path / "artifact.json",
            reader_factory=FakeExemplarReader,
        )

    source, target, manifest = _fixture(tmp_path / "confirmed")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["references"][0]["confirmed"] = False
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="human-confirmed"):
        run_handwriting_exemplars(
            source,
            target_crop=target,
            reference_manifest=manifest,
            references="reference-b",
            output_path=tmp_path / "artifact.json",
            reader_factory=FakeExemplarReader,
        )

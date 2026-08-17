import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from sibyl.experiments.handwriting_preprocess import (
    evaluate_candidate,
    run_handwriting_preprocess,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), (12, 34, 56)).save(source)
    crop = tmp_path / "region-02.png"
    Image.new("RGB", (10, 5), (100, 110, 120)).save(crop)
    compare = tmp_path / "trocr-compare.json"
    compare.write_text(
        json.dumps(
            {
                "experiment": "trocr_compare",
                "source": str(source),
                "regions": [
                    {
                        "region_id": "region-02",
                        "crop": {
                            "path": str(crop),
                            "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                            "width": 10,
                            "height": 5,
                            "source_bbox": {"left": 1, "top": 2, "right": 11, "bottom": 7},
                            "source_coordinate_space": "source",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, crop, compare


class FakeQwen:
    model = "qwen-test"
    calls: ClassVar[int] = 0
    observed_images: ClassVar[list[tuple[int, int]]] = []

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer

    def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        type(self).calls += 1
        self.observed_images.append(image.size)
        if type(self).calls == 2:
            response = {"message": {"content": "truncated"}}
            self.observer(response)
            return {"status": "truncated_response", "error": "truncated"}, 1.0
        response = {"message": {"content": "parsed"}}
        self.observer(response)
        return {"status": "ok", "text": "target word"}, 1.0

    def release(self) -> None:
        return None


class FakeTrOCR:
    calls: ClassVar[int] = 0

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        type(self).calls += 1
        if type(self).calls == 2:
            raise RuntimeError("fake failure")
        return "other word", 2.0


def _run(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    source, _, compare = _fixture(tmp_path)
    FakeQwen.calls = 0
    FakeQwen.observed_images = []
    FakeTrOCR.calls = 0
    return run_handwriting_preprocess(
        source,
        runs=2,
        output_path=tmp_path / "artifact.json",
        compare_artifact=compare,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (FakeTrOCR(), {"model": "trocr-test", "processor": "fake"}),
        **kwargs,
    )


def test_variants_are_ordered_deterministically_and_preserve_provenance(tmp_path: Path) -> None:
    artifact = _run(tmp_path, regions="region-02")
    region = artifact["regions"][0]
    names = [variant["variant"] for variant in region["variants"]]
    assert names == [
        "rgb-original",
        "grayscale",
        "rgb-2x",
        "rgb-3x",
        "grayscale-2x",
        "contrast-grayscale",
        "contrast-grayscale-2x",
    ]
    assert region["source_crop"]["source_bbox"] == {
        "left": 1,
        "top": 2,
        "right": 11,
        "bottom": 7,
    }
    assert (
        region["source_crop"]["sha256"]
        == hashlib.sha256(Path(region["source_crop"]["path"]).read_bytes()).hexdigest()
    )
    assert artifact["source_sha256"]


def test_variants_preserve_aspect_ratio_and_record_hashes(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    variants = artifact["regions"][0]["variants"]
    assert all(
        variant["dimensions"]["width"] / variant["dimensions"]["height"] == 2
        for variant in variants
    )
    assert all(Path(variant["path"]).is_file() for variant in variants)
    assert all(
        variant["sha256"] == hashlib.sha256(Path(variant["path"]).read_bytes()).hexdigest()
        for variant in variants
    )
    assert variants[0]["sha256"] == artifact["regions"][0]["source_crop"]["sha256"]


def test_repeated_reads_preserve_raw_and_truncated_responses(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    qwen = artifact["regions"][0]["variants"][0]["qwen"]
    assert len(qwen["runs"]) == 2
    assert qwen["runs"][0]["raw_response"] == {"message": {"content": "parsed"}}
    assert qwen["runs"][1]["status"] == "truncated_response"
    assert qwen["runs"][1]["raw_response"] == {"message": {"content": "truncated"}}
    trocr = artifact["regions"][0]["variants"][0]["trocr"]
    assert trocr["configuration"]["model"] == "trocr-test"
    assert any(read["status"] == "request_failure" for read in trocr["runs"])


def test_cross_variant_candidates_and_model_comparison_are_evidence_only(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    region = artifact["regions"][0]
    assert region["candidates"]["qwen"]["candidate"] == "target word"
    assert region["candidates"]["trocr"]["candidate"] == "other word"
    assert "target word" in region["comparison"]["qwen"]["stable_across_all_variants"]
    assert region["comparison"]["cross_model_agreement"] == []
    assert "ground_truth" not in region["candidates"]["qwen"]


def test_human_review_is_separate_and_evaluation_is_deterministic(tmp_path: Path) -> None:
    review = tmp_path / "review.yaml"
    review.write_text('ground_truth:\n  text: "target word"\n  confirmed: true\n', encoding="utf-8")
    artifact = _run(tmp_path, review_path=review)
    assert artifact["review"] == {"ground_truth": {"text": "target word", "confirmed": True}}
    assert artifact["regions"][0]["evaluation"]["qwen"]["normalized_exact_match"] is True
    assert evaluate_candidate("a b", "a c")["character_edit_distance"] == 1


def test_line_selection_takes_precedence_over_region_selection(tmp_path: Path) -> None:
    source, _crop, compare = _fixture(tmp_path)
    line = tmp_path / "line.png"
    Image.new("RGB", (8, 4), (1, 2, 3)).save(line)
    reread = tmp_path / "transcription-reread.json"
    reread.write_text(
        json.dumps(
            {
                "experiment": "transcription_reread",
                "source": str(source),
                "regions": [
                    {
                        "region_id": "region-02",
                        "line_localization": {
                            "regions": [
                                {
                                    "line_id": "region-02-line-04",
                                    "path": str(line),
                                    "source_bbox": {"left": 3, "top": 4, "right": 11, "bottom": 8},
                                    "source_coordinate_space": "source",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_handwriting_preprocess(
        source,
        runs=1,
        regions="region-02",
        lines="region-02-line-04",
        output_path=tmp_path / "artifact.json",
        compare_artifact=compare,
        reread_artifact=reread,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (FakeTrOCR(), {}),
    )
    assert [item["target_id"] for item in result["regions"]] == ["region-02-line-04"]

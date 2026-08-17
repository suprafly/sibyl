import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from sibyl.experiments.trocr_compare import run_compare_experiment


def _page(path: Path) -> Path:
    image = Image.new("RGB", (100, 100), (12, 34, 56))
    image.save(path)
    return path


class FakeLocalizer:
    model = "qwen-test"

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        response = {"message": {"content": "localization raw"}}
        self.observer(response)
        return {"text_regions": [{"bbox_2d": [100, 100, 900, 900]}]}, 1.0

    def release(self) -> None:
        return None


class FakeQwen:
    model = "qwen-test"
    images: ClassVar[list[Image.Image]] = []
    calls = 0

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer

    def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.images.append(image)
        type(self).calls += 1
        raw = {"message": {"content": f"raw-{self.calls}"}}
        self.observer(raw)
        return {"status": "ok", "text": "same reading" if self.calls % 2 else "other"}, 1.0

    def release(self) -> None:
        return None


class FakeTrOCR:
    images: ClassVar[list[Image.Image]] = []

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        self.images.append(image)
        return "trocr reading", 2.0


def test_compare_persists_one_rgb_crop_and_reuses_hash_and_pixels(tmp_path: Path) -> None:
    FakeQwen.images = []
    FakeQwen.calls = 0
    trocr = FakeTrOCR()
    artifact = run_compare_experiment(
        _page(tmp_path / "page.png"),
        runs=2,
        output_path=tmp_path / "artifact.json",
        localizer_factory=FakeLocalizer,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (trocr, {"processor": "fake", "model": "fake-trocr"}),
    )
    region = artifact["regions"][0]
    crop_path = Path(region["crop"]["path"])
    assert region["qwen_input_hash"] == region["trocr_input_hash"]
    assert region["crop"]["sha256"] == hashlib.sha256(crop_path.read_bytes()).hexdigest()
    with Image.open(crop_path) as crop:
        assert crop.mode == "RGB"
        assert crop.getpixel((0, 0)) == (12, 34, 56)
    assert len(FakeQwen.images) == len(trocr.images) == 2
    assert {id(image) for image in FakeQwen.images + trocr.images} == {id(FakeQwen.images[0])}
    assert len(region["qwen"]["runs"]) == len(region["trocr"]["runs"]) == 2
    assert region["qwen"]["runs"][0]["raw_response"] == {"message": {"content": "raw-1"}}


def test_compare_records_stability_and_overlap(tmp_path: Path) -> None:
    class StableQwen(FakeQwen):
        def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            self.images.append(image)
            self.observer({"raw": "qwen"})
            return {"status": "ok", "text": "same"}, 1.0

    artifact = run_compare_experiment(
        _page(tmp_path / "page.png"),
        runs=3,
        output_path=tmp_path / "artifact.json",
        localizer_factory=FakeLocalizer,
        reader_factory=StableQwen,
        trocr_factory=lambda: (FakeTrOCR(), {"model": "fake-trocr"}),
    )
    comparison = artifact["regions"][0]["comparison"]
    assert comparison["qwen_stable"] is True
    assert comparison["trocr_stable"] is True
    assert comparison["overlap"] == []


def test_trocr_unavailable_does_not_prevent_qwen(tmp_path: Path) -> None:
    artifact = run_compare_experiment(
        _page(tmp_path / "page.png"),
        runs=1,
        output_path=tmp_path / "artifact.json",
        localizer_factory=FakeLocalizer,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (_ for _ in ()).throw(RuntimeError("missing model")),
    )
    region = artifact["regions"][0]
    assert region["trocr"]["status"] == "model_unavailable"
    assert region["qwen"]["runs"][0]["status"] == "ok"


def test_explicit_region_selection_is_generic(tmp_path: Path) -> None:
    artifact = run_compare_experiment(
        _page(tmp_path / "page.png"),
        runs=1,
        regions="region-01",
        output_path=tmp_path / "artifact.json",
        localizer_factory=FakeLocalizer,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (FakeTrOCR(), {}),
    )
    assert [region["region_id"] for region in artifact["regions"]] == ["region-01"]


def test_region_selection_preserves_localization_indexes_after_duplicates(tmp_path: Path) -> None:
    class SparseLocalizer(FakeLocalizer):
        def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            boxes = [[100 + index * 20, 100, 110 + index * 20, 200] for index in range(5)]
            boxes.extend([boxes[0], boxes[0], boxes[0], boxes[0]])
            boxes.append([900, 100, 990, 200])
            return {"text_regions": [{"bbox_2d": box} for box in boxes]}, 1.0

    artifact = run_compare_experiment(
        _page(tmp_path / "page.png"),
        runs=1,
        regions="region-10",
        output_path=tmp_path / "artifact.json",
        localizer_factory=SparseLocalizer,
        reader_factory=FakeQwen,
        trocr_factory=lambda: (FakeTrOCR(), {}),
    )
    assert [region["region_id"] for region in artifact["regions"]] == ["region-10"]

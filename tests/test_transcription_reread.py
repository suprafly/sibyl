import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast

from PIL import Image

from sibyl.experiments.transcription_reread import (
    LOCALIZATION_NUM_PREDICT,
    LOCALIZATION_SCHEMA,
    REGIONAL_PROMPT,
    REGIONAL_SCHEMA,
    OllamaTextRegionLocalizer,
    deduplicate_regions,
    run_reread_experiment,
    validate_regions,
)
from sibyl.transform import map_prepared_bounds


class FakeLocalizer:
    model = "qwen3-vl:8b"

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer
        self.images: list[Image.Image] = []

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.images.append(image)
        raw = {"message": {"content": "localization raw"}}
        self.observer(raw)
        return {
            "text_regions": [{"bbox_2d": [100, 100, 300, 300]}, {"bbox_2d": [100, 100, 300, 300]}]
        }, 2.0

    def release(self) -> None:
        return None


class FakeReader:
    model = "qwen3-vl:8b"
    instances: ClassVar[list["FakeReader"]] = []

    def __init__(self, observer: Callable[[dict[str, Any]], None]) -> None:
        self.observer = observer
        self.images: list[Image.Image] = []
        self.instances.append(self)

    def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        self.images.append(image)
        raw = {"message": {"content": f"raw-{len(self.images)}"}}
        self.observer(raw)
        return {"text": "scion" if len(self.images) < 3 else "stem"}, 3.0

    def release(self) -> None:
        return None


def page(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), (12, 34, 56)).save(path)
    return path


def test_bbox_validation_rejects_malformed_range_inversion_and_zero_area() -> None:
    accepted, rejected = validate_regions(
        [
            {"bbox_2d": [1, 2, 3, 4]},
            {"bbox_2d": [1, 2, 3]},
            {"bbox_2d": [-1, 2, 3, 4]},
            {"bbox_2d": [4, 2, 3, 4]},
            {"bbox_2d": [1, 2, 1, 4]},
            {"bbox_2d": [1, 2, 3, float("inf")]},
        ]
    )
    assert [item["bbox_2d"] for item in accepted] == [[1.0, 2.0, 3.0, 4.0]]
    assert len(rejected) == 5


def test_localizer_has_dedicated_controls_and_minimal_request(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "text_regions": [
                                    {"bbox_2d": [1, 2, 3, 4]},
                                    {"bbox_2d": [10, 20, 30, 40]},
                                    {"bbox_2d": [100, 200, 300, 400]},
                                    {"bbox_2d": [450, 500, 600, 700]},
                                    {"bbox_2d": [750, 800, 900, 950]},
                                ]
                            }
                        )
                    }
                },
            ).encode()

    def urlopen(request: Any, timeout: int) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    localizer = OllamaTextRegionLocalizer(model="test", base_url="http://test")
    result, _duration = localizer.localize(Image.new("RGB", (4, 4)))
    payload = json.loads(captured["request"].data)
    assert len(result["text_regions"]) == 5
    assert payload["options"]["num_predict"] == LOCALIZATION_NUM_PREDICT == 512
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["keep_alive"] == 0
    assert payload["format"] == LOCALIZATION_SCHEMA
    assert captured["timeout"] == 300


def test_truncated_localization_is_preserved_and_classified(monkeypatch: Any) -> None:
    body = {
        "done_reason": "length",
        "eval_count": 256,
        "message": {
            "content": json.dumps(
                {"text_regions": [{"bbox_2d": [99, 102, 289, 154, 1000, 1000, 0, 0]}]}
            )
        },
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(body).encode()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, timeout: Response(),
    )
    result, _duration = OllamaTextRegionLocalizer(model="test", base_url="http://test").localize(
        Image.new("RGB", (4, 4))
    )
    assert result["status"] == "truncated_response"
    assert result["text_regions"][0]["bbox_2d"][-1] == 0
    assert result["raw_response"] == body


def test_localization_schema_requires_exactly_four_values() -> None:
    properties = cast(dict[str, Any], LOCALIZATION_SCHEMA["properties"])
    text_regions = cast(dict[str, Any], properties["text_regions"])
    items = cast(dict[str, Any], text_regions["items"])
    item_properties = cast(dict[str, Any], items["properties"])
    bbox_schema = cast(
        dict[str, Any],
        item_properties["bbox_2d"],
    )
    assert bbox_schema["minItems"] == bbox_schema["maxItems"] == 4


def test_deduplication_is_deterministic_for_duplicates_and_overlap() -> None:
    regions = [
        {"index": 0, "bbox_2d": [0.0, 0.0, 100.0, 100.0]},
        {"index": 1, "bbox_2d": [0.0, 0.0, 100.0, 100.0]},
        {"index": 2, "bbox_2d": [10.0, 10.0, 90.0, 90.0]},
        {"index": 3, "bbox_2d": [500.0, 500.0, 600.0, 600.0]},
    ]
    kept, rejected = deduplicate_regions(regions)
    assert [item["index"] for item in kept] == [0, 3]
    assert [item["index"] for item in rejected] == [1, 2]


def test_mapping_is_prepared_to_source_and_preserves_provenance() -> None:
    assert map_prepared_bounds((10, 20, 30, 40), (100, 200), (1000, 2000)).left == 100


def test_region_first_reuses_one_prepared_image_and_exact_rgb_crop(tmp_path: Path) -> None:
    FakeReader.instances = []
    localizers: list[FakeLocalizer] = []

    def localizer_factory(observer: Callable[[dict[str, Any]], None]) -> FakeLocalizer:
        value = FakeLocalizer(observer)
        localizers.append(value)
        return value

    result = run_reread_experiment(
        page(tmp_path),
        runs=5,
        output_path=tmp_path / "result.json",
        localizer_factory=localizer_factory,
        reader_factory=FakeReader,
    )
    assert len(localizers) == 1
    assert len(localizers[0].images) == 1
    assert len(result["regions"]) == 1
    region = result["regions"][0]
    assert region["model_coordinate_space"] == "qwen_0_1000"
    assert region["source_coordinate_space"] == "source"
    assert region["source_bbox"] == {"left": 9, "top": 9, "right": 31, "bottom": 31}
    assert region["width"] == region["height"] == 22
    assert Path(region["path"]).name == "region-01.png"
    with Image.open(region["path"]) as crop:
        assert crop.mode == "RGB"
        assert crop.getpixel((0, 0)) == (12, 34, 56)
    assert len(FakeReader.instances[0].images) == 5
    assert len({id(image) for image in FakeReader.instances[0].images}) == 1
    assert [read["status"] for read in region["reads"]] == ["ok"] * 5
    assert region["distinct_readings"] == ["scion", "stem"]
    assert region["stable"] is False
    assert region["reads"][0]["raw_response"] == {"message": {"content": "raw-1"}}


def test_invalid_and_failed_reads_are_not_empty_successes(tmp_path: Path) -> None:
    class BadReader(FakeReader):
        def read(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            self.images.append(image)
            self.observer({"raw": len(self.images)})
            if len(self.images) == 1:
                return {"status": "invalid_response", "error": "bad json"}, 1.0
            raise RuntimeError("request failed")

    result = run_reread_experiment(
        page(tmp_path),
        runs=2,
        output_path=tmp_path / "result.json",
        localizer_factory=FakeLocalizer,
        reader_factory=BadReader,
    )
    assert [read["status"] for read in result["regions"][0]["reads"]] == [
        "invalid_response",
        "failed",
    ]
    assert result["regions"][0]["reads"][0]["text"] is None


def test_schemas_and_prompt_are_minimal_and_isolated() -> None:
    assert REGIONAL_SCHEMA == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    assert "bbox" not in REGIONAL_SCHEMA["properties"]
    assert "candidate" not in REGIONAL_PROMPT.lower()
    assert "text_regions" in cast(dict[str, Any], LOCALIZATION_SCHEMA["properties"])

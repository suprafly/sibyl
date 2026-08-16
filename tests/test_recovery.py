import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from sibyl.recovery import (
    OllamaPageInterpreter,
    RegionBounds,
    format_recovery,
    map_prepared_bounds,
    prepare_vlm_image,
    recover_page,
)


class FakeInterpreter:
    model = "fake-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.mode == "L"
        assert image.size == (100, 100)
        return {
            "page_interpretation": {"layout": "heading and note"},
            "regions": [
                {
                    "order": 1,
                    "kind": "heading",
                    "left": 0.1,
                    "top": 0.1,
                    "right": 0.9,
                    "bottom": 0.4,
                    "text": "Calamodin",
                }
            ],
        }, 4.0

    def release(self) -> None:
        pass


class FakeRecognizer:
    def recognize(self, image: Image.Image) -> tuple[str, float]:
        assert image.mode == "RGB"
        assert image.size == (80, 30)
        return "Calamondin", 2.5


def test_vlm_preparation_is_bounded_and_preserves_source() -> None:
    source = Image.new("RGB", (3900, 5200), "white")
    source.putpixel((0, 0), (1, 2, 3))
    prepared, dimensions = prepare_vlm_image(source)
    assert dimensions == (1536, 2048)
    assert prepared.mode == "L"
    assert source.size == (3900, 5200)
    assert source.getpixel((0, 0)) == (1, 2, 3)


def test_prepared_dimensions_scale_and_coordinate_mapping_are_deterministic() -> None:
    source_size = (3900, 5200)
    prepared_size = (1536, 2048)
    assert prepared_size[0] / source_size[0] == prepared_size[1] / source_size[1]
    assert map_prepared_bounds((327, 707, 887, 875), prepared_size, source_size).left == 830
    assert map_prepared_bounds((327, 707, 887, 875), prepared_size, source_size) == RegionBounds(
        830, 1795, 2252, 2222
    )


class _Response(BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _ollama_response(message: dict[str, Any]) -> _Response:
    body = {"message": message, "prompt_eval_count": 3159, "eval_count": 42}
    return _Response(json.dumps(body).encode())


def test_qwen_structured_json_is_accepted_from_content(monkeypatch: Any) -> None:
    payload = {"regions": [{"kind": "text", "order": 0, "bbox_2d": [1, 2, 10, 20]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    interpreter = OllamaPageInterpreter(model="test")
    result, _ = interpreter.interpret(Image.new("L", (20, 20)))
    assert result == payload
    assert interpreter.response_metadata["prompt_tokens"] == 3159


def test_qwen_structured_json_is_accepted_from_thinking(monkeypatch: Any) -> None:
    payload = {"figures": [{"label": "diagram", "bbox_2d": [1, 2, 10, 20]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response(
            {"content": "", "thinking": json.dumps(payload)}
        ),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["figures"] == payload["figures"]
    assert result["regions"][0]["kind"] == "figure"


def test_qwen_invalid_structured_output_is_explicit(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": "not json", "thinking": "{}"}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert result["raw_response"]["message"]["content"] == "not json"


class FigureInterpreter:
    model = "fake-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.size == (1536, 2048)
        return {
            "page_interpretation": {},
            "regions": [
                {"order": 0, "kind": "figure", "text": "diagram", "bbox_2d": [327, 707, 887, 875]}
            ],
        }, 14_393.5

    def release(self) -> None:
        pass


class FailingRecognizer:
    def recognize(self, image: Image.Image) -> tuple[str, float]:
        raise AssertionError("figure regions must not be sent to TrOCR")


def test_figure_crops_use_mapped_original_coordinates_and_record_benchmark(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    page = recover_page(image_path, FigureInterpreter(), FailingRecognizer())
    region = page.regions[0]
    assert region.prepared_bounds == RegionBounds(327, 707, 887, 875)
    assert region.bounds == RegionBounds(830, 1795, 2252, 2222)
    crop = Path(region.source["crop"])
    assert crop.exists()
    assert Image.open(crop).size == (1422, 427)
    benchmark = page.runtime["benchmark"]
    assert benchmark["preparation_dimensions"] == {"width": 1536, "height": 2048}
    assert benchmark["qwen_ms"] == 14393.5
    assert benchmark["region_count"] == 1
    assert benchmark["trocr_timings"] == []
    assert benchmark["total_recovery_ms"] > 0


def test_recovery_preserves_source_qwen_text_and_recognizer_evidence(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = recover_page(
        image_path,
        FakeInterpreter(),
        FakeRecognizer(),
        recognizer_metadata={"device": "cpu"},
    )
    assert page.dimensions == {"width": 100, "height": 100}
    assert page.regions[0].qwen_text == "Calamodin"
    assert page.regions[0].text == "Calamondin"
    assert page.regions[0].bounds.left == 10
    assert json.loads(format_recovery(page))["regions"][0]["source"]["image"] == str(image_path)

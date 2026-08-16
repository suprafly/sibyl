import json
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from sibyl.transform import (
    OllamaDrawingLocalizer,
    OllamaPageInterpreter,
    RegionBounds,
    format_text_transform,
    format_transform,
    map_prepared_bounds,
    pad_normalized_bounds,
    prepare_vlm_image,
    qwen_bbox_to_normalized,
    transform_page,
    write_markdown_transform,
)


class _Response(BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _ollama_response(message: dict[str, Any]) -> _Response:
    body = {"message": message, "prompt_eval_count": 3159, "eval_count": 42}
    return _Response(json.dumps(body).encode())


def test_vlm_preparation_is_bounded_and_preserves_source() -> None:
    source = Image.new("RGB", (3900, 5200), "white")
    source.putpixel((0, 0), (1, 2, 3))
    prepared, dimensions = prepare_vlm_image(source)
    assert dimensions == (1536, 2048)
    assert prepared.mode == "L"
    assert source.getpixel((0, 0)) == (1, 2, 3)


def test_prepared_coordinate_mapping_is_deterministic() -> None:
    assert qwen_bbox_to_normalized((330, 707, 887, 872)) == pytest.approx(
        (0.33, 0.707, 0.887, 0.872)
    )
    assert map_prepared_bounds((327, 707, 887, 875), (1536, 2048), (3900, 5200)) == RegionBounds(
        830, 1795, 2252, 2222
    )


def test_qwen_page_structured_text_is_accepted_from_content(monkeypatch: Any) -> None:
    payload = {"page_interpretation": {"text": ["faithful page text"]}}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    interpreter = OllamaPageInterpreter(model="test")
    result, _ = interpreter.interpret(Image.new("L", (20, 20)))
    assert result["page_text"] == ["faithful page text"]
    assert result["regions"] == []


def test_qwen_page_structured_text_is_accepted_from_thinking(monkeypatch: Any) -> None:
    payload = {"page_interpretation": {"text": ["thinking text"]}}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": "", "thinking": json.dumps(payload)}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["page_text"] == ["thinking text"]


def test_qwen_page_prompt_preserves_nearby_handwriting_and_excludes_graphics(
    monkeypatch: Any,
) -> None:
    requests: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: int) -> _Response:
        requests.append(json.loads(request.data))
        return _ollama_response(
            {"content": json.dumps({"page_interpretation": {"text": []}})}
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    request = requests[0]
    prompt = request["messages"][0]["content"]
    for phrase in (
        "reading order",
        "Preserve wording, spelling",
        "unfamiliar terminology",
        "Do not invent text",
        "[unclear]",
        "graphical arrows",
        "diagram strokes",
        "connectors",
        "Handwritten words remain text",
        "near a drawing",
    ):
        assert phrase in prompt
    assert "Scion" not in prompt
    assert "Splice" not in prompt
    assert "text_regions" not in request["format"]["properties"]["page_interpretation"][
        "properties"
    ]


def test_qwen_page_malformed_response_fails_explicitly(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps({"wrong": []})}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert "unsupported transform schema" in result["error"]


class PageInterpreter:
    model = "fake-page-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        return {
            "page_interpretation": {
                "text": [
                    "Xylem",
                    "- preserves unfamiliar terminology",
                    "Handwritten note near figure",
                ]
            },
            "page_text": [
                "Xylem",
                "- preserves unfamiliar terminology",
                "Handwritten note near figure",
            ],
        }, 11.0

    def release(self) -> None:
        pass


class DrawingLocalizer:
    model = "fake-drawing-qwen"
    response_metadata: ClassVar[dict[str, list[str]]] = {"response_fields": ["content"]}

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.size == (1536, 2048)
        return (
            {
                "drawings": [
                    {
                        "bbox_2d": [330, 707, 887, 872],
                        "model_bbox": [330, 707, 887, 872],
                        "bbox_coordinate_space": "qwen_0_1000",
                        "description": "complete figure evidence",
                    }
                ]
            },
            7.0,
        )

    def release(self) -> None:
        pass


def test_page_text_and_one_complete_figure_are_projected_without_text_assets(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    stale_assets = tmp_path / "page.sibyl" / "assets"
    stale_assets.mkdir(parents=True)
    Image.new("RGB", (2, 2), "black").save(stale_assets / "text-01.png")
    Image.new("RGB", (2, 2), "black").save(stale_assets / "figure-02.png")
    page = transform_page(image_path, PageInterpreter(), drawing_localizer=DrawingLocalizer())

    assert page.page_text == [
        "Xylem",
        "- preserves unfamiliar terminology",
        "Handwritten note near figure",
    ]
    assert len(page.regions) == 1
    assert page.regions[0].kind == "figure"
    assert page.regions[0].bounds == RegionBounds(1178, 3633, 3567, 4578)
    assert page.runtime["benchmark"]["drawing_regions"] == 1
    assert "text_localization_ms" not in page.runtime["benchmark"]
    assert "trocr_ms" not in page.runtime["benchmark"]
    assert "disagreements" not in page.runtime

    markdown = write_markdown_transform(page).read_text()
    assert "Xylem" in markdown
    assert "Handwritten note near figure" in markdown
    assert "![Figure 1](assets/figure-01.png)" in markdown
    assert len(list((tmp_path / "page.sibyl" / "assets").glob("figure-*.png"))) == 1
    assert not list((tmp_path / "page.sibyl" / "assets").glob("text-*.png"))

    artifact = json.loads((tmp_path / "page.sibyl" / "transform.json").read_text())
    assert artifact["page_text"] == page.page_text
    assert artifact["regions"][0]["source"]["provenance"] == ["drawing_localization"]


def test_page_text_projection_is_qwen_text_without_ocr_layers(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = transform_page(image_path, PageInterpreter(), drawing_localizer=None)
    assert format_text_transform(page) == "\n\n".join(page.page_text)
    structured = json.loads(format_transform(page))
    assert structured["page_text"] == page.page_text
    assert "recognizer" not in structured["runtime"]


def test_drawing_localizer_accepts_and_normalizes_qwen_bbox(monkeypatch: Any) -> None:
    payload = {"drawings": [{"bbox_2d": [330, 707, 887, 872], "description": "figure"}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["drawings"][0]["model_bbox"] == [330.0, 707.0, 887.0, 872.0]
    assert result["drawings"][0]["bbox_coordinate_space"] == "qwen_0_1000"


def test_drawing_localizer_accepts_thinking_json_and_bbox_alias(monkeypatch: Any) -> None:
    payload = {"drawings": [{"bbox": [200, 300, 700, 800]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response(
            {"content": "", "thinking": json.dumps(payload)}
        ),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["drawings"][0]["model_bbox"] == [200.0, 300.0, 700.0, 800.0]


def test_drawing_localizer_rejects_invalid_bbox(monkeypatch: Any) -> None:
    payload = {"drawings": [{"bbox_2d": [0, 0, 1, 1001]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert "invalid bbox" in result["error"]


def test_drawing_localizer_rejects_unsupported_entry(monkeypatch: Any) -> None:
    payload = {"drawings": [{"label": "not a drawing record"}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert result["unsupported_entries"] == payload["drawings"]


def test_drawing_padding_is_proportional_and_clamped() -> None:
    assert pad_normalized_bounds((0.0, 0.0, 0.2, 0.4)) == pytest.approx((0.0, 0.0, 0.21, 0.42))
    assert pad_normalized_bounds((0.9, 0.9, 1.0, 1.0)) == pytest.approx((0.895, 0.895, 1.0, 1.0))


def test_drawing_localization_failure_preserves_page_text(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FailingDrawing:
        model = "fake-drawing-qwen"

        def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
            return {"status": "failure", "error": "mock localization unavailable"}, 2.0

        def release(self) -> None:
            pass

    page = transform_page(image_path, PageInterpreter(), drawing_localizer=FailingDrawing())
    assert page.page_text[0] == "Xylem"
    assert page.regions == []
    assert page.runtime["drawing_localization"]["status"] == "failure"

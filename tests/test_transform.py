import json
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image, ImageDraw

from sibyl.transform import (
    OllamaDrawingLocalizer,
    OllamaPageInterpreter,
    RegionBounds,
    _content_bounds,
    format_text_transform,
    format_transform,
    map_prepared_bounds,
    pad_normalized_bounds,
    prepare_page_image,
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


@pytest.mark.parametrize(
    "configured,expected",
    [(None, (1536, 2048)), ("2048", (2048, 2731)), ("2560", (2560, 3413))],
)
def test_page_preparation_resolution_is_explicit_and_aspect_preserving(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected: tuple[int, int],
) -> None:
    if configured is None:
        monkeypatch.delenv("SIBYL_PAGE_MAX_DIMENSION", raising=False)
    else:
        monkeypatch.setenv("SIBYL_PAGE_MAX_DIMENSION", configured)
    prepared, dimensions = prepare_page_image(Image.new("RGB", (3900, 5200), "white"))
    assert dimensions == expected
    assert dimensions[0] / dimensions[1] == pytest.approx(3900 / 5200, abs=1e-4)
    assert prepared.size == expected


def test_page_resolution_does_not_change_drawing_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Image.new("RGB", (3900, 5200), "white")
    baseline, baseline_dimensions = prepare_vlm_image(source)
    monkeypatch.setenv("SIBYL_PAGE_MAX_DIMENSION", "2560")
    experimental, experimental_dimensions = prepare_vlm_image(source)
    assert experimental_dimensions == baseline_dimensions == (1536, 2048)
    assert experimental.size == baseline.size
    assert experimental.tobytes() == baseline.tobytes()


def test_page_focus_defaults_to_full_and_content_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(source).rectangle((50, 100, 349, 499), fill=(20, 20, 20))

    monkeypatch.delenv("SIBYL_PAGE_FOCUS", raising=False)
    full, full_dimensions = prepare_page_image(source)
    monkeypatch.setenv("SIBYL_PAGE_FOCUS", "content")
    focused, focused_dimensions = prepare_page_image(source)

    assert full_dimensions == (400, 600)
    assert full.size == full_dimensions
    assert _content_bounds(source) == (50, 100, 350, 500)
    assert focused.size == focused_dimensions
    assert focused_dimensions == (300, 400)
    assert max(focused_dimensions) <= 1536
    assert focused_dimensions[0] / focused_dimensions[1] == pytest.approx(0.75)


def test_content_focus_stays_one_page_and_preserves_drawing_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "page.png"
    source = Image.new("RGB", (3900, 5200), "white")
    ImageDraw.Draw(source).rectangle((500, 1000, 3499, 4499), fill=(20, 20, 20))
    source.save(image_path)
    monkeypatch.setenv("SIBYL_PAGE_FOCUS", "content")
    page = transform_page(image_path, PageInterpreter(), drawing_localizer=DrawingLocalizer())

    assert page.runtime["page_transform"]["page_focus"] == "content"
    assert page.runtime["benchmark"]["page_focus"] == "content"
    page_dimensions = page.runtime["benchmark"]["page_preparation_dimensions"]
    assert page_dimensions["width"] <= 1536
    assert page_dimensions["height"] <= 1536
    assert page.runtime["benchmark"]["drawing_preparation_dimensions"] == {
        "width": 1536,
        "height": 2048,
    }
    assert len(page.regions) == 1


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
    assert prompt == (
        "Transform this handwritten page. Return only JSON matching the schema. "
        "Transcribe the ordinary handwritten notes and textual marks visible on the page "
        "in reading "
        "order. Read the actual handwriting and preserve the wording, spelling, "
        "capitalization, punctuation, shorthand, symbols that are genuinely part "
        "of written text, and unfamiliar terminology. Do not autocorrect, replace "
        "a word with a semantically more likely word, normalize unfamiliar words, "
        "or invent text. Use [unclear] only when the letterforms are genuinely "
        "unreadable. Do not transcribe graphical elements of drawings or diagrams "
        "as page text, including arrows, diagram strokes, lines, and graphical "
        "connectors that clearly function as graphics. A handwritten word remains "
        "text even when it is physically near a drawing; do not exclude text merely "
        "because it is beside, above, below, or adjacent to a figure. Do not "
        "enumerate spatial text regions, drawings, or invent coordinates. Do not "
        "perform exhaustive text localization; a separate pass handles drawing "
        "localization."
    )
    assert request["options"]["num_predict"] == 256
    assert request["think"] is False
    assert request["stream"] is False
    assert request["keep_alive"] == 0
    for phrase in (
        "reading order",
        "preserve the wording, spelling",
        "unfamiliar terminology",
        "or invent text",
        "[unclear]",
        "including arrows",
        "diagram strokes",
        "connectors",
        "A handwritten word remains text",
        "physically near a drawing",
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


def test_experimental_page_resolution_is_recorded_without_changing_drawing_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    monkeypatch.setenv("SIBYL_PAGE_MAX_DIMENSION", "2560")
    page = transform_page(image_path, PageInterpreter(), drawing_localizer=DrawingLocalizer())
    benchmark = page.runtime["benchmark"]
    assert benchmark["page_preparation_dimensions"] == {"width": 2560, "height": 3413}
    assert benchmark["drawing_preparation_dimensions"] == {"width": 1536, "height": 2048}


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

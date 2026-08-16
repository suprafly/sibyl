import json
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from sibyl.transform import (
    OllamaDrawingLocalizer,
    OllamaPageInterpreter,
    OllamaTextLocalizer,
    RegionBounds,
    format_transform,
    map_prepared_bounds,
    pad_normalized_bounds,
    prepare_vlm_image,
    qwen_bbox_to_normalized,
    transform_page,
    write_markdown_transform,
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
        lambda request, timeout: _ollama_response({"content": "", "thinking": json.dumps(payload)}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["figures"] == payload["figures"]
    assert result["regions"][0]["kind"] == "figure"


def test_qwen_page_interpretation_shape_from_thinking_is_normalized(monkeypatch: Any) -> None:
    expected_text = [
        "Xylem",
        "- transports mineral nutrients and water from root to stem",
        "Phloem",
        "- transports food and nutrients from leaves to storage organs.",
        "Sapling grafting - what we will do now.",
        "N -> H -> Wurd",
    ]
    expected_description = (
        "Simplified representation of a plant stem with upward arrows indicating "
        "growth direction (N) and connection to a grafting point (H) and 'Wurd' "
        "label pointing to the grafted section."
    )
    payload = {
        "page_interpretation": {
            "text": expected_text,
            "diagram": [
                {
                    "bbox": [0.329, 0.717, 0.427, 0.874],
                    "description": expected_description,
                }
            ],
        }
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": "", "thinking": json.dumps(payload)}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (1536, 2048)))
    assert result["page_text"] == expected_text
    assert len(result["regions"]) == 1
    assert result["regions"][0]["kind"] == "figure"
    assert result["regions"][0]["bbox_2d"] == [505, 1468, 656, 1790]
    assert result["regions"][0]["text"] == expected_description
    assert map_prepared_bounds((505, 1468, 656, 1790), (1536, 2048), (3900, 5200)) == RegionBounds(
        1282, 3727, 1666, 4545
    )


def test_qwen_page_prompt_is_not_responsible_for_spatial_text(monkeypatch: Any) -> None:
    requests: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: int) -> _Response:
        requests.append(json.loads(request.data))
        return _ollama_response(
            {"content": json.dumps({"page_interpretation": {"text": [], "drawing": []}})}
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    request = requests[0]
    assert "regions" not in request["format"]["properties"]
    assert "handwritten notes" in request["messages"][0]["content"]
    assert "spatial text regions" not in request["messages"][0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text_regions": [{"bbox_2d": [110, 120, 290, 160]}]},
        {
            "text_regions": [
                {"bbox_2d": [110, 120, 290, 160]},
                {"bbox_2d": [145, 165, 960, 305]},
            ]
        },
        {"text_regions": []},
    ],
)
def test_qwen_text_localizer_accepts_minimal_bbox_only_responses(
    monkeypatch: Any, payload: dict[str, Any]
) -> None:
    requests: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: int) -> _Response:
        requests.append(json.loads(request.data))
        return _ollama_response({"content": json.dumps(payload)})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result, _ = OllamaTextLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["text_regions"] == payload["text_regions"]
    schema = requests[0]["format"]
    assert list(schema["properties"]) == ["text_regions"]
    assert list(schema["properties"]["text_regions"]["items"]["properties"]) == ["bbox_2d"]
    prompt = requests[0]["messages"][0]["content"]
    assert "Return only the requested JSON." in prompt
    assert "transcrib" not in prompt.lower()
    assert "reading order" not in prompt.lower()


def test_qwen_text_localizer_rejects_malformed_response_explicitly(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response(
            {"content": json.dumps({"text_regions": [{"bbox_2d": [1, 2, 3]}]})}
        ),
    )
    result, _ = OllamaTextLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert "unsupported text localization schema" in result["error"]


def test_qwen_page_prompt_preserves_text_near_figures_and_excludes_graphics(
    monkeypatch: Any,
) -> None:
    requests: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: int) -> _Response:
        requests.append(json.loads(request.data))
        return _ollama_response(
            {
                "content": json.dumps(
                    {
                        "page_interpretation": {
                            "text": [
                                "Xylem",
                                "- transports mineral nutrients and water from root to Scion",
                                "Phloem",
                                "- transports food and nutrients from leaves to storage organs.",
                                "Splice grafting - what we will do now.",
                            ]
                        }
                    }
                )
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    prompt = requests[0]["messages"][0]["content"]
    for instruction in (
        "ordinary handwritten notes",
        "preserve the wording, spelling",
        "unfamiliar terminology",
        "autocorrect",
        "semantically more likely word",
        "[unclear]",
        "graphical elements of drawings or diagrams",
        "arrows, diagram strokes, lines",
        "graphical connectors",
        "handwritten word remains text",
        "physically near a drawing",
        "beside, above, below, or adjacent",
    ):
        assert instruction in prompt
    assert "Splice" not in prompt
    assert "Scion" not in prompt
    assert result["page_text"] == [
        "Xylem",
        "- transports mineral nutrients and water from root to Scion",
        "Phloem",
        "- transports food and nutrients from leaves to storage organs.",
        "Splice grafting - what we will do now.",
    ]
    assert "↓" not in result["page_text"]
    assert "→" not in result["page_text"]
    assert "←" not in result["page_text"]
    assert "urds" not in result["page_text"]


def test_qwen_invalid_structured_output_is_explicit(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": "not json", "thinking": "{}"}),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert result["raw_response"]["message"]["content"] == "not json"


def test_qwen_valid_but_unsupported_json_is_distinguished(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response(
            {"content": json.dumps({"page_interpretation": {"unknown": []}})}
        ),
    )
    result, _ = OllamaPageInterpreter(model="test").interpret(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert "valid JSON but unsupported transform schema" in result["error"]
    assert "page_interpretation" in result["error"]


def test_drawing_localizer_accepts_qwen_0_1000_content_json_and_uses_dedicated_schema(
    monkeypatch: Any,
) -> None:
    requests: list[dict[str, Any]] = []
    payload = {"drawings": [{"bbox_2d": [100, 200, 800, 900], "description": "complete figure"}]}

    def urlopen(request: Any, timeout: int) -> _Response:
        requests.append(json.loads(request.data))
        return _ollama_response({"content": json.dumps(payload)})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    localizer = OllamaDrawingLocalizer(model="test")
    result, _ = localizer.localize(Image.new("L", (1536, 2048)))
    assert result == {
        "drawings": [
            {
                "bbox_2d": [100.0, 200.0, 800.0, 900.0],
                "model_bbox": [100.0, 200.0, 800.0, 900.0],
                "bbox_coordinate_space": "qwen_0_1000",
                "description": "complete figure",
            }
        ]
    }
    assert requests[0]["format"]["required"] == ["drawings"]
    assert "ordinary handwriting" in requests[0]["messages"][0]["content"]
    assert "complete figure" in requests[0]["messages"][0]["content"]


def test_drawing_localizer_accepts_thinking_json(monkeypatch: Any) -> None:
    payload = {"drawings": [{"bbox_2d": [0, 0, 1000, 1000]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": "", "thinking": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result == {
        "drawings": [
            {
                "bbox_2d": [0.0, 0.0, 1000.0, 1000.0],
                "model_bbox": [0.0, 0.0, 1000.0, 1000.0],
                "bbox_coordinate_space": "qwen_0_1000",
            }
        ]
    }


def test_drawing_localizer_normalizes_established_bbox_alias(monkeypatch: Any) -> None:
    payload = {
        "drawings": [{"bbox": [200, 300, 700, 800], "description": "existing fixture shape"}]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result == {
        "drawings": [
            {
                "bbox_2d": [200.0, 300.0, 700.0, 800.0],
                "model_bbox": [200.0, 300.0, 700.0, 800.0],
                "bbox_coordinate_space": "qwen_0_1000",
                "description": "existing fixture shape",
            }
        ]
    }


def test_drawing_localizer_accepts_observed_qwen_0_1000_fixture(
    monkeypatch: Any,
) -> None:
    payload = {
        "drawings": [
            {
                "bbox_2d": [330, 707, 887, 872],
                "description": (
                    "Hand-drawn schematic of three vertical elements connected by arrows, "
                    "with a labeled arrow pointing to the rightmost element"
                ),
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (1536, 2048)))
    drawing = result["drawings"][0]
    assert drawing["model_bbox"] == [330.0, 707.0, 887.0, 872.0]
    assert drawing["bbox_coordinate_space"] == "qwen_0_1000"
    assert drawing["description"] == payload["drawings"][0]["description"]


def test_drawing_localizer_accepts_qwen_0_1000_bbox_alias(monkeypatch: Any) -> None:
    payload = {"drawings": [{"bbox": [330, 707, 887, 872]}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (1536, 2048)))
    assert result["drawings"][0]["bbox_coordinate_space"] == "qwen_0_1000"
    assert result["drawings"][0]["model_bbox"] == [330.0, 707.0, 887.0, 872.0]


@pytest.mark.parametrize(
    "bbox",
    ([0, -1, 10, 20], [0, 0, 1001, 1000], [10, 20, 10, 21], [0, 0, 0, 1]),
)
def test_drawing_localizer_rejects_invalid_coordinate_ranges(
    monkeypatch: Any, bbox: list[int]
) -> None:
    payload = {"drawings": [{"bbox_2d": bbox}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (1536, 2048)))
    assert result["status"] == "failure"
    assert "invalid bbox" in result["error"]


def test_drawing_localizer_reports_valid_json_with_unsupported_entry(monkeypatch: Any) -> None:
    payload = {"drawings": [{"label": "not a supported drawing record"}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _ollama_response({"content": json.dumps(payload)}),
    )
    result, _ = OllamaDrawingLocalizer(model="test").localize(Image.new("L", (20, 20)))
    assert result["status"] == "failure"
    assert "valid drawing localization JSON" in result["error"]
    assert "drawing entry 0" in result["error"]
    assert '"label": "not a supported drawing record"' in result["error"]
    assert result["unsupported_entries"] == payload["drawings"]


def test_drawing_localizer_distinguishes_missing_and_malformed_json(monkeypatch: Any) -> None:
    responses = iter(
        (
            _ollama_response({"content": json.dumps({})}),
            _ollama_response({"content": "not json"}),
        )
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: next(responses))
    image = Image.new("L", (20, 20))
    missing = OllamaDrawingLocalizer(model="test").localize(image)[0]
    malformed = OllamaDrawingLocalizer(model="test").localize(image)[0]
    assert "valid drawing localization JSON" in missing["error"]
    assert "top-level drawings" in missing["error"]
    assert "no valid drawing localization JSON" in malformed["error"]


def test_drawing_padding_is_proportional_and_clamped() -> None:
    assert pad_normalized_bounds((0.0, 0.0, 0.2, 0.4)) == pytest.approx((0.0, 0.0, 0.21, 0.42))
    assert pad_normalized_bounds((0.9, 0.9, 1.0, 1.0)) == pytest.approx((0.895, 0.895, 1.0, 1.0))


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


class PageLevelTextInterpreter:
    model = "fake-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        return {
            "page_interpretation": {"text": ["Xylem", "N -> H -> H"]},
            "page_text": ["Xylem", "N -> H -> H"],
            "regions": [
                {
                    "order": index,
                    "kind": "figure",
                    "text": f"diagram {index + 1}",
                    "bbox_2d": list(bbox),
                }
                for index, bbox in enumerate(
                    (
                        (499, 1468, 581, 1788),
                        (817, 1509, 911, 1788),
                        (1055, 1481, 1143, 1788),
                    )
                )
            ],
        }, 1.0

    def release(self) -> None:
        pass


def test_page_level_text_does_not_trigger_trocr_or_fake_spatial_regions(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    page = transform_page(image_path, PageLevelTextInterpreter(), FailingRecognizer())
    assert page.page_text == ["Xylem", "N -> H -> H"]
    assert len(page.regions) == 3
    assert all(region.kind == "figure" for region in page.regions)
    assert page.runtime["benchmark"]["spatial_text_regions"] == 0
    assert page.runtime["benchmark"]["drawing_regions"] == 3
    assert page.runtime["benchmark"]["trocr_attempts"] == 0
    assert page.runtime["benchmark"]["trocr_timings"] == []
    assert page.runtime["disagreements"] == []
    assert page.runtime["recognizer"]["status"] == "not_applicable"
    assert [Path(region.source["crop"]).name for region in page.regions] == [
        "figure-01.png",
        "figure-02.png",
        "figure-03.png",
    ]
    assert [region.bounds for region in page.regions] == [
        RegionBounds(1267, 3727, 1475, 4540),
        RegionBounds(2074, 3831, 2313, 4540),
        RegionBounds(2679, 3760, 2902, 4540),
    ]
    markdown = write_markdown_transform(page).read_text()
    assert all(f"assets/figure-{index:02d}.png" in markdown for index in range(1, 4))


def test_figure_crops_use_mapped_original_coordinates_and_record_benchmark(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    page = transform_page(image_path, FigureInterpreter(), FailingRecognizer())
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
    assert benchmark["total_transform_ms"] > 0


def test_transform_preserves_source_qwen_text_and_recognizer_evidence(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = transform_page(
        image_path,
        FakeInterpreter(),
        FakeRecognizer(),
        recognizer_metadata={"device": "cpu"},
    )
    assert page.dimensions == {"width": 100, "height": 100}
    assert page.regions[0].qwen_text == "Calamodin"
    assert page.regions[0].text == "Calamondin"
    assert page.regions[0].bounds.left == 10
    assert json.loads(format_transform(page))["regions"][0]["source"]["image"] == str(image_path)


class PageOnlyInterpreter:
    model = "fake-page-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        return {
            "page_interpretation": {"text": ["Exact page text"]},
            "page_text": ["Exact page text"],
            "regions": [],
        }, 11.0

    def release(self) -> None:
        pass


class FakeDrawingLocalizer:
    model = "fake-drawing-qwen"

    def __init__(self, result: dict[str, Any], timing: float = 7.0) -> None:
        self.result = result
        self.timing = timing
        self.response_metadata = {"response_fields": ["thinking"]}

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.size == (100, 100)
        return self.result, self.timing

    def release(self) -> None:
        pass


class QwenDrawingLocalizer:
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
                        "description": "model evidence",
                    }
                ]
            },
            7.0,
        )

    def release(self) -> None:
        pass


def test_dedicated_localization_preserves_text_descriptions_coordinates_and_markdown(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    source = Image.new("RGB", (100, 100), "white")
    source.putpixel((50, 50), (255, 0, 0))
    source.save(image_path)
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        FailingRecognizer(),
        drawing_localizer=FakeDrawingLocalizer(
            {
                "drawings": [
                    {"bbox_2d": [0.4, 0.4, 0.6, 0.6], "description": "model evidence"},
                    {"bbox_2d": [0.0, 0.0, 0.1, 0.1]},
                ]
            }
        ),
    )
    assert page.page_text == ["Exact page text"]
    assert len(page.regions) == 2
    first = page.regions[0]
    assert first.normalized_bounds == (0.4, 0.4, 0.6, 0.6)
    assert first.source["provenance"] == ["drawing_localization"]
    assert first.source["drawing_localization"]["description"] == "model evidence"
    assert first.bounds == RegionBounds(39, 39, 61, 61)
    assert Image.open(first.source["crop"]).size == (22, 22)
    benchmark = page.runtime["benchmark"]
    assert benchmark["page_transform_ms"] == 11.0
    assert benchmark["drawing_localization_ms"] == 7.0
    assert benchmark["crop_ms"] >= 0
    assert benchmark["trocr_attempts"] == 0
    assert page.runtime["page_transform"]["status"] == "success"
    assert page.runtime["drawing_localization"]["status"] == "success"
    markdown = write_markdown_transform(page).read_text()
    assert "Exact page text" in markdown
    assert "![Figure 1](assets/figure-01.png)" in markdown
    assert "model evidence" not in markdown


def test_qwen_bbox_maps_and_crops_original_source_for_markdown(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        FailingRecognizer(),
        drawing_localizer=QwenDrawingLocalizer(),
    )
    region = page.regions[0]
    assert region.normalized_bounds == pytest.approx(
        (0.33, 0.707, 0.887, 0.872)
    )
    assert region.prepared_bounds == RegionBounds(464, 1431, 1405, 1803)
    assert region.bounds == RegionBounds(1178, 3633, 3567, 4578)
    assert region.source["model_bbox"] == [330, 707, 887, 872]
    assert region.source["bbox_coordinate_space"] == "qwen_0_1000"
    assert Image.open(region.source["crop"]).size == (2389, 945)
    assert write_markdown_transform(page).read_text() == (
        "Exact page text\n\n![Figure 1](assets/figure-01.png)\n"
    )


def test_qwen_bbox_conversion_is_not_literal_prepared_pixel_interpretation() -> None:
    assert qwen_bbox_to_normalized((330, 707, 887, 872)) == pytest.approx(
        (0.33, 0.707, 0.887, 0.872)
    )
    assert qwen_bbox_to_normalized((330, 707, 887, 872)) != pytest.approx(
        (330 / 1536, 707 / 2048, 887 / 1536, 872 / 2048)
    )


def test_drawing_localization_failure_preserves_successful_page_text(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        FailingRecognizer(),
        drawing_localizer=FakeDrawingLocalizer(
            {
                "status": "failure",
                "error": "mock localization unavailable",
                "unsupported_entries": [{"unexpected": "shape"}],
            }
        ),
    )
    assert page.page_text == ["Exact page text"]
    assert page.regions == []
    assert page.runtime["drawing_localization"]["status"] == "failure"
    assert "unavailable" in page.runtime["drawing_localization"]["error"]
    assert page.runtime["drawing_localization"]["unsupported_entries"] == [{"unexpected": "shape"}]
    assert page.runtime["benchmark"]["drawing_regions"] == 0


def test_zero_drawings_is_a_successful_text_only_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        FailingRecognizer(),
        drawing_localizer=FakeDrawingLocalizer({"drawings": []}),
    )
    assert page.page_text == ["Exact page text"]
    assert page.regions == []
    assert page.runtime["drawing_localization"]["status"] == "success"
    assert page.runtime["benchmark"]["drawing_regions"] == 0
    assert not list((tmp_path / "page.sibyl" / "assets").glob("text-*.png"))


class SpatialTextInterpreter:
    model = "fake-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.size == (1536, 2048)
        return {
            "page_interpretation": {"text": ["page-level evidence"]},
            "page_text": ["page-level evidence"],
            "regions": [
                {
                    "order": 2,
                    "kind": "text",
                    "bbox_2d": [500, 500, 700, 650],
                    "text": "qwen second",
                },
                {
                    "order": 1,
                    "kind": "text",
                    "bbox_2d": [100, 100, 300, 250],
                    "text": "qwen first",
                },
            ],
        }, 11.0

    def release(self) -> None:
        pass


class MinimalTextLocalizer:
    model = "fake-text-qwen"

    def __init__(self, bboxes: list[list[int]], timing: float = 3.0) -> None:
        self.bboxes = bboxes
        self.timing = timing
        self.response_metadata = {"response_fields": ["content"]}

    def localize(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.size in {(1536, 2048), (100, 100)}
        return {"text_regions": [{"bbox_2d": bbox} for bbox in self.bboxes]}, self.timing

    def release(self) -> None:
        pass


def test_minimal_bbox_response_is_ordered_and_composed_with_trocr(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    bboxes = [
        [145, 405, 970, 530],
        [110, 120, 290, 160],
        [105, 555, 880, 680],
        [145, 165, 960, 305],
        [110, 345, 300, 400],
    ]
    recognizer = RecordingRecognizer(["one", "two", "three", "four", "five"])
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        recognizer,
        text_localizer=MinimalTextLocalizer(bboxes),
        recognizer_metadata={"device": "cpu"},
    )
    assert [region.order for region in page.regions] == [1, 2, 3, 4, 5]
    assert [region.text for region in page.regions] == ["one", "two", "three", "four", "five"]
    assert [region.source["model_bbox"] for region in page.regions] == [
        [110, 120, 290, 160],
        [145, 165, 960, 305],
        [110, 345, 300, 400],
        [145, 405, 970, 530],
        [105, 555, 880, 680],
    ]
    assert page.runtime["benchmark"]["spatial_text_regions"] == 5
    assert page.runtime["benchmark"]["trocr_attempts"] == 5
    assert page.runtime["text_localization"]["status"] == "success"
    assert page.runtime["benchmark"]["text_localization_ms"] == 3.0
    assets = sorted((tmp_path / "page.sibyl" / "assets").glob("text-*.png"))
    assert [path.name for path in assets] == [
        "text-01.png",
        "text-02.png",
        "text-03.png",
        "text-04.png",
        "text-05.png",
    ]


def test_zero_bbox_response_does_not_attempt_trocr_or_create_full_page_crop(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = transform_page(
        image_path,
        PageOnlyInterpreter(),
        FailingRecognizer(),
        text_localizer=MinimalTextLocalizer([]),
    )
    assert page.page_text == ["Exact page text"]
    assert page.regions == []
    assert page.runtime["benchmark"]["spatial_text_regions"] == 0
    assert page.runtime["benchmark"]["trocr_attempts"] == 0
    assert not list((tmp_path / "page.sibyl" / "assets").glob("text-*.png"))


class RecordingRecognizer:
    def __init__(self, texts: list[str], fail_at: int | None = None) -> None:
        self.texts = texts
        self.fail_at = fail_at
        self.images: list[Image.Image] = []

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        index = len(self.images)
        self.images.append(image.copy())
        if index == self.fail_at:
            raise RuntimeError("mock TrOCR failure")
        return self.texts[index], 2.0


def test_spatial_text_uses_ordered_trocr_on_padded_original_resolution_crops(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    source = Image.new("RGB", (3900, 5200), "white")
    source.putpixel((500, 500), (255, 0, 0))
    source.save(image_path)
    recognizer = RecordingRecognizer(["first faithful", "second faithful"])
    page = transform_page(
        image_path,
        SpatialTextInterpreter(),
        recognizer,
        recognizer_metadata={"device": "cpu"},
    )
    assert [region.order for region in page.regions] == [1, 2]
    assert [region.text for region in page.regions] == ["first faithful", "second faithful"]
    assert page.page_text == ["first faithful", "second faithful"]
    assert all(image.width > 300 for image in recognizer.images)
    assert all(image.height > 300 for image in recognizer.images)
    assert all(region.source["bbox_coordinate_space"] == "qwen_0_1000" for region in page.regions)
    assert page.runtime["benchmark"]["spatial_text_regions"] == 2
    assert page.runtime["benchmark"]["trocr_attempts"] == 2
    assert page.runtime["benchmark"]["trocr_successes"] == 2
    assert page.runtime["benchmark"]["trocr_failures"] == 0
    assert page.runtime["qwen_page_text"] == ["page-level evidence"]
    assert page.runtime["disagreements"] == [
        {"order": 1, "qwen": "qwen first", "trocr": "first faithful", "status": "success"},
        {"order": 2, "qwen": "qwen second", "trocr": "second faithful", "status": "success"},
    ]
    markdown = write_markdown_transform(page).read_text()
    assert "page-level evidence" not in markdown
    assert "assets/text-01.png" not in markdown
    assets = sorted((tmp_path / "page.sibyl" / "assets").glob("text-*.png"))
    assert [asset.name for asset in assets] == ["text-01.png", "text-02.png"]
    artifact = json.loads((tmp_path / "page.sibyl" / "transform.json").read_text())
    first = artifact["regions"][0]["source"]
    assert first["text_crop"] == "assets/text-01.png"
    assert first["crop"]["source_bounds"] == [
        page.regions[0].bounds.left,
        page.regions[0].bounds.top,
        page.regions[0].bounds.right,
        page.regions[0].bounds.bottom,
    ]
    assert first["crop"]["width"] == Image.open(assets[0]).width
    assert first["crop"]["height"] == Image.open(assets[0]).height
    assert Image.open(assets[0]).convert("RGB").tobytes() == recognizer.images[0].tobytes()
    assert first["padding"]["normalized_proportion"] == pytest.approx(0.02)


def test_spatial_trocr_failure_is_explicit_and_does_not_use_qwen_as_canonical(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (3900, 5200), "white").save(image_path)
    recognizer = RecordingRecognizer(["unused", "second faithful"], fail_at=0)
    page = transform_page(
        image_path,
        SpatialTextInterpreter(),
        recognizer,
        recognizer_metadata={"device": "cpu"},
    )
    first = page.regions[0]
    assert first.text == ""
    assert first.qwen_text == "qwen first"
    assert first.recognizer["status"] == "failure"
    assert "mock TrOCR failure" in first.recognizer["error"]
    assert page.runtime["benchmark"]["trocr_successes"] == 1
    assert page.runtime["benchmark"]["trocr_failures"] == 1
    assert page.runtime["disagreements"][0]["qwen"] == "qwen first"
    artifact = json.loads(format_transform(page))
    assert artifact["regions"][0]["qwen_text"] == "qwen first"
    assert artifact["regions"][0]["recognizer"]["status"] == "failure"
    write_markdown_transform(page)
    assert (tmp_path / "page.sibyl" / "assets" / "text-01.png").exists()

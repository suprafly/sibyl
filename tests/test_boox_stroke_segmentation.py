from pathlib import Path
from typing import Any

from PIL import Image

from sibyl.experiments import boox_stroke_segmentation as segmentation


def stroke(
    identifier: str, order: int, left: float, top: float, right: float, bottom: float
) -> dict[str, Any]:
    return {
        "stroke_id": identifier,
        "order": order,
        "point_count": 10,
        "native_points": [{"x": left, "y": top}],
        "native_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
    }


def test_stroke_groups_are_ordered_and_preserve_geometry() -> None:
    groups = segmentation.derive_boox_groups(
        [
            stroke("b", 1, 40, 110, 60, 140),
            stroke("a", 0, 10, 10, 30, 40),
            stroke("c", 2, 70, 110, 90, 140),
        ],
        max_vertical_gap=10,
        max_word_gap=20,
        min_strokes=2,
    )
    assert [group["stroke_ids"] for group in groups["lines"]] == [["b", "c"]]
    assert groups["lines"][0]["source_bbox"] == {
        "left": 40.0,
        "top": 110.0,
        "right": 90.0,
        "bottom": 140.0,
    }
    assert groups["rejected"][0]["reason"] == "line_below_minimum"


def test_native_raster_mapping_accepts_exact_page() -> None:
    mapping = segmentation._native_raster_mapping((1404, 1872))
    assert mapping["native_dimensions"] == {"width": 1404, "height": 1872}
    assert mapping["raster_dimensions"] == {"width": 1404, "height": 1872}
    assert mapping["scale_x"] == mapping["scale_y"] == 1.0
    assert mapping["uniform_scaling"] is True


def test_native_raster_mapping_supports_proportional_downscale_and_upscale() -> None:
    down = segmentation._native_raster_mapping((702, 936))
    up = segmentation._native_raster_mapping((2808, 3744))
    assert down["scale_x"] == down["scale_y"] == 0.5
    assert up["scale_x"] == up["scale_y"] == 2.0


def test_native_raster_mapping_rejects_incompatible_aspect_ratio() -> None:
    try:
        segmentation._native_raster_mapping((1000, 1000))
    except ValueError as error:
        assert "aspect ratio" in str(error)
    else:
        raise AssertionError("incompatible page aspect ratio was accepted")


def test_native_bbox_mapping_is_deterministic() -> None:
    mapping = segmentation._native_raster_mapping((702, 936))
    native_bbox = {"left": 100.0, "top": 200.0, "right": 500.0, "bottom": 800.0}
    expected = {"left": 50.0, "top": 100.0, "right": 250.0, "bottom": 400.0}
    assert segmentation._map_bbox(native_bbox, mapping) == expected
    assert segmentation._map_bbox(native_bbox, mapping) == expected


def test_group_mapping_preserves_native_and_raster_provenance() -> None:
    mapping = segmentation._native_raster_mapping((702, 936))
    group = {
        "group_id": "boox-line-001",
        "native_bbox": {"left": 10.0, "top": 20.0, "right": 30.0, "bottom": 40.0},
    }
    mapped = segmentation._map_group(group, mapping)
    assert mapped["native_bbox"] == group["native_bbox"]
    assert mapped["raster_bbox"] == {"left": 5.0, "top": 10.0, "right": 15.0, "bottom": 20.0}
    assert mapped["source_bbox"] == mapped["raster_bbox"]


def test_horizontal_gap_splits_words_and_overlapping_strokes_stay_together() -> None:
    groups = segmentation.derive_boox_groups(
        [
            stroke("a", 0, 10, 10, 20, 30),
            stroke("b", 1, 22, 10, 32, 30),
            stroke("c", 2, 90, 10, 100, 30),
            stroke("d", 3, 91, 11, 102, 31),
        ],
        max_vertical_gap=5,
        max_word_gap=20,
        min_strokes=2,
    )
    assert len(groups["lines"]) == 1
    assert [word["stroke_ids"] for word in groups["words"]] == [["a", "b"], ["c", "d"]]


def test_figure_overlap_and_missing_bbox_are_rejected() -> None:
    groups = segmentation.derive_boox_groups(
        [
            stroke("text-1", 0, 10, 10, 30, 30),
            stroke("text-2", 1, 32, 10, 52, 30),
            stroke("figure-1", 2, 100, 100, 120, 120),
            stroke("figure-2", 3, 122, 100, 130, 120),
            {"stroke_id": "missing", "order": 4, "point_count": 1},
        ],
        max_vertical_gap=5,
        figure_regions=[{"bounds": {"left": 90, "top": 90, "right": 130, "bottom": 130}}],
    )
    assert [group["stroke_ids"] for group in groups["lines"]] == [["text-1", "text-2"]]
    assert {item["reason"] for item in groups["rejected"]} == {
        "overlaps_figure",
        "missing_or_empty_bbox",
    }


def test_source_crop_is_original_raster_and_deterministic(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), "white")
    image.putpixel((20, 20), (0, 0, 0))
    group = {
        "group_id": "boox-line-001",
        "source_bbox": {"left": 20.0, "top": 20.0, "right": 21.0, "bottom": 21.0},
        "stroke_ids": ["a"],
    }
    first = segmentation._crop_group(image, group, tmp_path / "first", padding=2)
    second = segmentation._crop_group(image, group, tmp_path / "second", padding=2)
    assert first["crop_sha256"] == second["crop_sha256"]
    assert first["crop_bbox"] == {"left": 18, "top": 18, "right": 23, "bottom": 23}


def test_markdown_metrics_and_unresolved_are_deterministic() -> None:
    metrics = segmentation.evaluate_markdown("Xylem\n⟦unresolved⟧", ["Xylem", "Phloem"])
    assert metrics["exact_line_match"] == 1
    assert metrics["normalized_line_match"] == 1
    assert metrics["unresolved_count"] == 1
    assert metrics["resolved_block_count"] == 1


def test_mocked_recognition_preserves_one_run_as_unresolved(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (10, 10), "white").save(crop)

    class FakeReader:
        model = "qwen-test"

        def read(
            self, images: list[Image.Image], prompt: str, controls: dict[str, Any]
        ) -> tuple[dict[str, Any], float]:
            assert len(images) == 1
            return {
                "status": "ok",
                "text": "candidate",
                "raw_response": {"message": {"content": "candidate"}},
            }, 0.0

        def release(self) -> None:
            pass

    def factory(observer: Any) -> FakeReader:
        return FakeReader()

    results, markdown = segmentation._recognize(
        [{"group_id": "line-1", "crop_path": str(crop)}],
        runs=1,
        num_predict=32,
        num_ctx=128,
        reader_factory=factory,
    )
    assert markdown == ["⟦unresolved⟧"]
    assert results[0]["analysis"]["readings"] == ["candidate"]
    assert results[0]["analysis"]["runs"][0]["raw_response"]["message"]["content"] == "candidate"

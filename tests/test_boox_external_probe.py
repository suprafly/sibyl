import struct
from pathlib import Path

from PIL import Image

from sibyl.experiments.boox_external_probe import probe_points, render_probe_strokes

PAGE_ID = "11111111-1111-1111-1111-111111111111"
POINT_ID = "22222222-2222-2222-2222-222222222222"
STROKE_ID = "33333333-3333-3333-3333-333333333333"


def _fixture(version: int = 1) -> bytes:
    point = struct.pack(">ffBBHI", 10.0, 20.0, 1, 2, 2048, 7)
    stroke_offset, stroke_size = 76, 4 + len(point)
    index_start = stroke_offset + stroke_size
    entry = STROKE_ID.encode() + stroke_offset.to_bytes(4, "big") + stroke_size.to_bytes(4, "big")
    return (
        version.to_bytes(4, "big")
        + PAGE_ID.encode()
        + POINT_ID.encode()
        + b"\0" * 4
        + point
        + entry
        + index_start.to_bytes(4, "big")
    )


def test_public_layout_probe_matches_fixture_and_preserves_point_fields() -> None:
    result = probe_points(
        _fixture(),
        page_id=PAGE_ID,
        point_resource_id=POINT_ID,
        page_size=(100.0, 100.0),
        shape_ids=[STROKE_ID],
    )
    assert result["status"] == "matched"
    assert result["stroke_count"] == 1
    assert result["point_count"] == 1
    assert result["strokes"][0]["shape_association"] is True
    assert result["strokes"][0]["points"][0]["pressure"] == 2048
    assert result["strokes"][0]["points"][0]["time_delta"] == 7


def test_public_layout_probe_reports_wrong_version_and_bounds() -> None:
    result = probe_points(
        _fixture(version=2),
        page_id=PAGE_ID,
        point_resource_id=POINT_ID,
        page_size=(5.0, 5.0),
        shape_ids=[STROKE_ID],
    )
    assert result["status"] == "partially_matches"
    assert any("version" in failure for failure in result["failures"])
    assert any("out-of-page" in failure for failure in result["failures"])


def test_probe_rendering_is_deterministic(tmp_path: Path) -> None:
    result = probe_points(
        _fixture(),
        page_id=PAGE_ID,
        point_resource_id=POINT_ID,
        page_size=(100.0, 100.0),
        shape_ids=[STROKE_ID],
    )
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    render_probe_strokes(first, (100, 100), result["strokes"])
    render_probe_strokes(second, (100, 100), result["strokes"])
    assert first.read_bytes() == second.read_bytes()
    assert Image.open(first).size == (100, 100)


def test_public_layout_probe_rejects_truncated_resource() -> None:
    result = probe_points(
        b"short",
        page_id=PAGE_ID,
        point_resource_id=POINT_ID,
        page_size=(100.0, 100.0),
    )
    assert result["status"] == "does_not_match"

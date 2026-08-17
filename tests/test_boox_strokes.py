import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from sibyl.experiments.boox_strokes import inspect_boox_strokes, inspect_boox_strokes_all


def _fixture(path: Path) -> None:
    page = "page-four"
    note_info = (
        b'{"pageNameList":["page-one","page-four"],"pageInfoMap":'
        b'{"page-one":{"width":100,"height":200},"page-four":{"width":100,"height":200}}}'
    )
    shape = b"shape record 11111111-1111-1111-1111-111111111111"
    import struct

    point = b"\0\0\0\1" + page.encode().ljust(32, b"x") + b"1".ljust(32, b"y") + b"\0\0\0\0"
    point += struct.pack(">ff", 10.0, 20.0) + b"\0" * 8
    point += struct.pack(">ff", 30.0, 40.0) + b"\0" * 8
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("root/note/pb/note_info", note_info)
        archive.writestr(f"root/shape/{page}#11111111-1111-1111-1111-111111111111#time", shape)
        archive.writestr(
            f"root/point/{page}/{page}#11111111-1111-1111-1111-111111111111#points", point
        )


def test_fixture_extracts_ordered_pages_and_preserves_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "fixture.note"
    _fixture(note)
    monkeypatch.chdir(tmp_path)
    result = inspect_boox_strokes(note, page=2, output=tmp_path / "out")
    assert [page["page_id"] for page in result["page_order"]] == ["page-one", "page-four"]
    assert result["selected_page"]["page_id"] == "page-four"
    assert all(len(resource["sha256"]) == 64 for resource in result["resources"])
    assert result["strokes"][0]["point_count"] == 2


def test_fixture_reconstruction_is_deterministic_and_native_sized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "fixture.note"
    _fixture(note)
    monkeypatch.chdir(tmp_path)
    first = inspect_boox_strokes(note, page=2, output=tmp_path / "one")
    second = inspect_boox_strokes(note, page=2, output=tmp_path / "two")
    assert first["strokes"] == second["strokes"]
    assert Image.open(tmp_path / "one/page-002-native.png").size == (100, 200)
    assert json.loads((tmp_path / "one/page-002-metadata.json").read_text()) == first


def test_uncertain_point_resource_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "fixture.note"
    _fixture(note)
    monkeypatch.chdir(tmp_path)
    result = inspect_boox_strokes(note, page=2, output=tmp_path / "out")
    assert result["raw_resource_preservation"] is True
    assert any(resource["kind"] == "point" for resource in result["resources"])


def test_all_pages_writes_deterministic_summary_and_continues_after_page_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "fixture.note"
    _fixture(note)
    monkeypatch.chdir(tmp_path)
    first = inspect_boox_strokes_all(note, output=tmp_path / "one")
    second = inspect_boox_strokes_all(note, output=tmp_path / "two")

    assert first == second
    assert first["page_count"] == 2
    assert first["pages"] == [
        {
            "page": 1,
            "page_id": "page-one",
            "status": "failure",
            "stroke_count": 0,
            "point_count": 0,
        },
        {
            "page": 2,
            "page_id": "page-four",
            "status": "success",
            "stroke_count": 1,
            "point_count": 2,
        },
    ]
    assert first["total_strokes"] == 1
    assert first["total_points"] == 2
    assert first["failures"][0]["page"] == 1
    assert (tmp_path / "one/page-002-metadata.json").is_file()
    assert json.loads((tmp_path / "one/../boox-strokes.json").read_text()) == first


def test_page_and_all_pages_are_mutually_exclusive() -> None:
    from sibyl.cli import build_parser

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["experiment", "boox-strokes", "note.note", "--page", "4", "--all-pages"]
        )
    assert error.value.code == 2

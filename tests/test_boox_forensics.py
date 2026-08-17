import json
import struct
import zipfile
from pathlib import Path

import pytest

from sibyl.experiments.boox_forensics import (
    _coordinate_hypotheses,
    parse_wire_message,
    write_page_forensics,
)


def test_wire_parser_preserves_wire_types_and_nested_boundaries() -> None:
    data = b"\x08\x96\x01\x15\x01\x02\x03\x04\x1a\x02\x08\x01"
    parsed = parse_wire_message(data)
    assert parsed is not None
    fields, end = parsed
    assert end == len(data)
    assert [field["wire_type"] for field in fields] == ["varint", "fixed32", "length_delimited"]
    assert fields[0]["value"] == 150
    assert fields[1]["raw_hex"] == "01 02 03 04"
    assert fields[2]["nested"][0]["field"] == 1
    assert fields[2]["payload_offset"] == 10


def test_wire_parser_rejects_truncated_and_invalid_fields() -> None:
    assert parse_wire_message(b"\x08") is None
    assert parse_wire_message(b"\x00") is None
    assert parse_wire_message(b"\x1a\x05\x08\x01") is None
    fixed64 = parse_wire_message(b"\x09\x01\x02\x03\x04\x05\x06\x07\x08")
    assert fixed64 is not None
    assert fixed64[0][0]["wire_type"] == "fixed64"


def test_coordinate_hypothesis_reports_bounds_without_accepting_semantics() -> None:
    data = b"\0" * 80
    data += struct.pack(">ff", 10.0, 20.0) + b"\0" * 8
    data += struct.pack(">ff", 200.0, 300.0) + b"\0" * 8
    hypothesis = _coordinate_hypotheses(data, 100.0, 100.0)[0]
    assert hypothesis["records_examined"] == 2
    assert hypothesis["records_in_page_bounds"] == 1
    assert hypothesis["accepted"] is False


def test_forensics_writes_byte_preserving_resources(tmp_path: Path) -> None:
    note_info = (
        b'{"pageNameList":["page-four"],"pageInfoMap":'
        b'{"page-four":{"width":100,"height":200}}}'
    )
    point = b"point-bytes"
    shape = b"shape-bytes"
    note = tmp_path / "sample.note"
    with zipfile.ZipFile(note, "w") as archive:
        archive.writestr("root/note/pb/note_info", note_info)
        archive.writestr("root/shape/page-four#shape-id#stamp", shape)
        archive.writestr("root/point/page-four/page-four#point-id#points", point)
    result = write_page_forensics(note, page=1, output=tmp_path / "out")
    assert (tmp_path / "out/page-001-point-resource.bin").read_bytes() == point
    assert (tmp_path / "out/page-001-shape-resource.bin").read_bytes() == shape
    assert result["semantic_interpretation"]["accepted_candidate"] is None
    saved = json.loads((tmp_path / "out/page-001-forensics.json").read_text())
    assert saved["resources"]["point"]["size"] == len(point)


def test_forensics_requires_one_shape_and_point_resource(tmp_path: Path) -> None:
    note = tmp_path / "sample.note"
    with zipfile.ZipFile(note, "w") as archive:
        archive.writestr(
            "root/note/pb/note_info",
            b'{"pageNameList":["page-four"],"pageInfoMap":{"page-four":{"width":100,"height":200}}}',
        )
    with pytest.raises(ValueError, match="exactly one"):
        write_page_forensics(note, page=1, output=tmp_path / "out")

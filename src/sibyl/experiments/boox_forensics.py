"""Schema-free forensic inspection of one BOOX page's binary resources."""

# The forensic JSON intentionally mirrors compact binary records.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

from sibyl.corpus import read_boox_metadata

WIRE_NAMES = {0: "varint", 1: "fixed64", 2: "length_delimited", 5: "fixed32"}


def _hex(data: bytes) -> str:
    return data.hex(" ")


def _hex_preview(data: bytes, *, limit: int = 256) -> str:
    if len(data) <= limit:
        return _hex(data)
    return f"{_hex(data[:limit])} ..."


def _varint(data: bytes, offset: int, end: int) -> tuple[int, int] | None:
    value = 0
    shift = 0
    cursor = offset
    while cursor < end and shift <= 63:
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
    return None


def parse_wire_message(
    data: bytes, *, start: int = 0, end: int | None = None, depth: int = 0
) -> tuple[list[dict[str, Any]], int] | None:
    """Parse a complete protobuf wire stream without assigning field semantics."""
    if depth > 8:
        return None
    limit = len(data) if end is None else end
    cursor = start
    fields: list[dict[str, Any]] = []
    while cursor < limit:
        field_start = cursor
        key_result = _varint(data, cursor, limit)
        if key_result is None:
            return None
        key, cursor = key_result
        field_number, wire_type = key >> 3, key & 7
        if field_number == 0 or wire_type not in WIRE_NAMES:
            return None
        node: dict[str, Any] = {
            "field": field_number,
            "wire_type": WIRE_NAMES[wire_type],
            "wire_type_number": wire_type,
            "offset": field_start,
            "key_length": cursor - field_start,
        }
        if wire_type == 0:
            result = _varint(data, cursor, limit)
            if result is None:
                return None
            value, cursor = result
            node.update({"value": value, "length": cursor - field_start})
        elif wire_type == 1:
            if cursor + 8 > limit:
                return None
            payload = data[cursor : cursor + 8]
            cursor += 8
            node.update({"length": 8, "raw_hex": _hex(payload), "fixed64_le": struct.unpack("<Q", payload)[0]})
        elif wire_type == 5:
            if cursor + 4 > limit:
                return None
            payload = data[cursor : cursor + 4]
            cursor += 4
            node.update({"length": 4, "raw_hex": _hex(payload), "fixed32_le": struct.unpack("<I", payload)[0]})
        else:
            length_result = _varint(data, cursor, limit)
            if length_result is None:
                return None
            length, payload_start = length_result
            payload_end = payload_start + length
            if payload_end > limit:
                return None
            payload = data[payload_start:payload_end]
            cursor = payload_end
            node.update(
                {
                    "length": length,
                    "payload_offset": payload_start,
                    "raw_hex": _hex_preview(payload),
                    "raw_hex_truncated": len(payload) > 256,
                }
            )
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
            if text and all(character.isprintable() for character in text):
                node["utf8"] = text
            nested = parse_wire_message(data, start=payload_start, end=payload_end, depth=depth + 1)
            if nested is not None and nested[1] == payload_end:
                node["nested"] = nested[0]
        node["end_offset"] = cursor
        fields.append(node)
    return fields, cursor


def scan_wire_candidates(data: bytes, *, minimum_fields: int = 2) -> list[dict[str, Any]]:
    """Find complete wire streams embedded in an unknown resource.

    This is a framing scan only. Results are candidates, never semantic points.
    """
    candidates: list[dict[str, Any]] = []
    for offset in range(len(data)):
        parsed = parse_wire_message(data, start=offset)
        if parsed is None:
            continue
        fields, end = parsed
        if len(fields) < minimum_fields or end - offset < 4:
            continue
        candidates.append(
            {
                "offset": offset,
                "length": end - offset,
                "field_count": len(fields),
                "fields": fields,
                "raw_hex_context": _hex_preview(
                    data[max(0, offset - 8) : min(len(data), end + 8)]
                ),
                "raw_hex_context_truncated": end - offset + 16 > 256,
            }
        )
    return candidates


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resource_for_page(note_path: Path, page_id: str) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]], list[tuple[str, bytes]]]:
    with zipfile.ZipFile(note_path) as archive:
        entries = [(name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")]
    page_entries = [(name, data) for name, data in entries if page_id in name]
    shapes = [(name, data) for name, data in page_entries if "/shape/" in name]
    points = [(name, data) for name, data in page_entries if "/point/" in name and name.endswith("#points")]
    return page_entries, shapes, points


def inspect_page_forensics(note_path: Path, *, page: int, output: Path) -> dict[str, Any]:
    note = read_boox_metadata(note_path)
    if page < 1 or page > note["page_count"]:
        raise ValueError(f"page {page} is outside the note")
    selected = note["pages"][page - 1]
    _page_entries, shapes, points = _resource_for_page(note_path, selected["page_id"])
    if len(points) != 1 or len(shapes) != 1:
        raise ValueError("page forensics requires exactly one page-local shape and point resource")
    point_name, point_data = points[0]
    shape_name, shape_data = shapes[0]
    output.mkdir(parents=True, exist_ok=True)
    point_path = output / f"page-{page:03d}-point-resource.bin"
    shape_path = output / f"page-{page:03d}-shape-resource.bin"
    point_path.write_bytes(point_data)
    shape_path.write_bytes(shape_data)
    point_id = Path(point_name).name.split("#")[1]
    shape_id = Path(shape_name).name.split("#")[1]
    candidates = scan_wire_candidates(point_data)
    shape_member: str | None = None
    shape_payload = shape_data
    try:
        with zipfile.ZipFile(io.BytesIO(shape_data)) as shape_archive:
            shape_member = shape_archive.namelist()[0]
            shape_payload = shape_archive.read(shape_member)
    except (OSError, zipfile.BadZipFile, IndexError):
        pass
    shape_candidates = scan_wire_candidates(shape_payload)
    shape_stream = parse_wire_message(shape_payload)
    point_tail = point_data[80:] if len(point_data) >= 80 else b""
    result: dict[str, Any] = {
        "artifact": "boox_strokes_forensics",
        "page": page,
        "page_id": selected["page_id"],
        "page_dimensions": {"width": selected["width"], "height": selected["height"]},
        "source_note": str(note_path),
        "source_note_sha256": _sha256(note_path.read_bytes()),
        "resources": {
            "point": {"id": point_id, "path": point_name, "size": len(point_data), "sha256": _sha256(point_data)},
            "shape": {"id": shape_id, "path": shape_name, "size": len(shape_data), "sha256": _sha256(shape_data)},
        },
        "raw_files": {"point": str(point_path), "shape": str(shape_path)},
        "wire_format": {
            "classification": "protobuf-like framing investigated; schema unproven",
            "supported_wire_types": WIRE_NAMES,
            "point_complete_stream_candidates": candidates[:64],
            "shape_complete_stream_candidates": shape_candidates[:1],
            "point_candidate_count": len(candidates),
            "shape_candidate_count": len(shape_candidates),
            "shape_container": {
                "outer_format": "ZIP" if shape_member else "not_detected",
                "member": shape_member,
                "inner_size": len(shape_payload),
                "inner_sha256": _sha256(shape_payload),
                "complete_stream": shape_stream is not None,
                "top_level_field_count": len(shape_stream[0]) if shape_stream else 0,
            },
        },
        "observed_layout": {
            "point_prefix_length_considered": min(80, len(point_data)),
            "point_tail_length": len(point_tail),
            "point_tail_mod_16": len(point_tail) % 16,
            "point_header": {
                "leading_u32_be": int.from_bytes(point_data[:4], "big") if len(point_data) >= 4 else None,
                "page_id_ascii": point_data[4:40].decode("ascii", errors="replace") if len(point_data) >= 40 else None,
                "point_resource_id_ascii": point_data[40:76].decode("ascii", errors="replace") if len(point_data) >= 76 else None,
                "reserved_prefix_hex": _hex(point_data[76:80]) if len(point_data) >= 80 else None,
            },
            "ascii_uuid_matches_in_shape": [uid for uid in _uuids(shape_data) if uid in {point_id, shape_id}],
        },
        "shape_records_observed": _shape_record_summary(shape_stream, shape_payload),
        "coordinate_hypotheses": _coordinate_hypotheses(point_data, selected["width"], selected["height"]),
        "semantic_interpretation": {
            "coordinate_fields": [],
            "coordinate_system": "undetermined",
            "decoded_points": 0,
            "decoded_strokes": 0,
            "accepted_candidate": None,
            "confidence": "insufficient evidence",
        },
        "uncertainties": [
            "The resource header and payload do not establish a schema by themselves.",
            "No coordinate field is labeled or accepted from wire framing alone.",
            "Shape-to-point semantic association remains unresolved; IDs are preserved exactly.",
        ],
    }
    return result


def _uuids(data: bytes) -> list[str]:
    import re

    pattern = re.compile(rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    return [match.decode("ascii").lower() for match in pattern.findall(data)]


def _shape_record_summary(parsed: tuple[list[dict[str, Any]], int] | None, data: bytes) -> dict[str, Any]:
    if parsed is None:
        return {"repeated_field": None, "record_count": 0, "records": []}
    records: list[dict[str, Any]] = []
    for field in parsed[0]:
        if field["field"] != 1 or field["wire_type"] != "length_delimited":
            continue
        nested = field.get("nested", [])
        record: dict[str, Any] = {"offset": field["offset"], "length": field["length"], "fields": {}}
        for child in nested:
            value: Any = child.get("value", child.get("utf8", child.get("raw_hex")))
            record["fields"].setdefault(str(child["field"]), []).append(value)
        records.append(record)
    return {"repeated_field": 1, "record_count": len(records), "records": records}


def _coordinate_hypotheses(data: bytes, width: float, height: float) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for name, offset, x_offset, y_offset in (
        ("float32_be_xy_at_80", 80, 0, 4),
        ("u32_then_float32_be_xy_at_800", 800, 4, 8),
    ):
        total = max(0, (len(data) - offset) // 16)
        in_bounds = 0
        for index in range(total):
            cursor = offset + index * 16
            x, y = struct.unpack(">ff", data[cursor + x_offset : cursor + x_offset + 8])
            if 0 <= x <= width and 0 <= y <= height:
                in_bounds += 1
        hypotheses.append(
            {
                "name": name,
                "offset": offset,
                "record_stride": 16,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "records_examined": total,
                "records_in_page_bounds": in_bounds,
                "accepted": False,
                "reason": "plausible numeric layout requires stroke-boundary and shape-range validation",
            }
        )
    return hypotheses


def write_page_forensics(note_path: Path, *, page: int, output: Path) -> dict[str, Any]:
    result = inspect_page_forensics(note_path, page=page, output=output)
    path = output / f"page-{page:03d}-forensics.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

"""Experimental validation of the public boox-note-parser point layout."""

# The probe records compact binary evidence fields.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

EXPECTED_HEADER_SIZE = 76
POINT_SIZE = 16
INDEX_ENTRY_SIZE = 44


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big")


def _bounds(points: list[dict[str, Any]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}


def probe_points(
    data: bytes,
    *,
    page_id: str,
    point_resource_id: str,
    page_size: tuple[float, float],
    shape_ids: list[str] | None = None,
) -> dict[str, Any]:
    width, height = page_size
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    if len(data) < EXPECTED_HEADER_SIZE + 4:
        failures.append("resource is shorter than header and index pointer")
        return _result("does_not_match", evidence, failures)
    observed_page_id = data[4:40].decode("ascii", errors="replace")
    observed_point_id = data[40:76].decode("ascii", errors="replace")
    evidence.extend(
        [
            {
                "offset": 0,
                "expected": "u32 big-endian version/page value",
                "observed_hex": data[:4].hex(" "),
                "decoded_value": _u32(data, 0),
                "confidence": "observed header field",
            },
            {
                "offset": 4,
                "expected": f"page ID {page_id}",
                "observed_hex": data[4:40].hex(" "),
                "decoded_value": observed_page_id,
                "confidence": "exact match" if observed_page_id == page_id else "mismatch",
            },
            {
                "offset": 40,
                "expected": f"point resource ID {point_resource_id}",
                "observed_hex": data[40:76].hex(" "),
                "decoded_value": observed_point_id,
                "confidence": "exact match" if observed_point_id == point_resource_id else "mismatch",
            },
        ]
    )
    if _u32(data, 0) != 1:
        failures.append("header version/page value is not the observed version 1")
    if observed_page_id != page_id:
        failures.append("page ID does not match selected page")
    if observed_point_id != point_resource_id:
        failures.append("point resource ID does not match selected resource")
    index_start = _u32(data, len(data) - 4)
    evidence.append(
        {
            "offset": len(data) - 4,
            "expected": "absolute big-endian index start offset",
            "observed_hex": data[-4:].hex(" "),
            "decoded_value": index_start,
            "confidence": "structurally validated if entries consume to EOF",
        }
    )
    if index_start < EXPECTED_HEADER_SIZE or index_start > len(data) - 4:
        failures.append("index start offset is outside the resource")
        return _result("does_not_match", evidence, failures)
    index_bytes = len(data) - 4 - index_start
    if index_bytes % INDEX_ENTRY_SIZE:
        failures.append("index bytes are not divisible into 44-byte entries")
        return _result("partially_matches", evidence, failures)
    entries: list[dict[str, Any]] = []
    strokes: list[dict[str, Any]] = []
    shape_set = set(shape_ids or [])
    for entry_offset in range(index_start, len(data) - 4, INDEX_ENTRY_SIZE):
        stroke_id = data[entry_offset : entry_offset + 36].decode("ascii", errors="replace")
        stroke_offset = _u32(data, entry_offset + 36)
        stroke_size = _u32(data, entry_offset + 40)
        entry: dict[str, Any] = {
            "index_offset": entry_offset,
            "stroke_id": stroke_id,
            "stroke_offset": stroke_offset,
            "stroke_size": stroke_size,
        }
        entries.append(entry)
        if stroke_offset + stroke_size > len(data) or stroke_size < 4:
            failures.append(f"stroke {stroke_id} points outside the resource")
            continue
        padding = data[stroke_offset : stroke_offset + 4]
        if padding != b"\0" * 4:
            failures.append(f"stroke {stroke_id} lacks documented 4-byte zero padding")
        point_bytes = stroke_size - 4
        if point_bytes % POINT_SIZE:
            failures.append(f"stroke {stroke_id} size is not 4 + N*16")
            continue
        points: list[dict[str, Any]] = []
        for point_offset in range(stroke_offset + 4, stroke_offset + stroke_size, POINT_SIZE):
            x, y, tilt_x, tilt_y, pressure, time_delta = struct.unpack(
                ">ffBBHI", data[point_offset : point_offset + POINT_SIZE]
            )
            points.append(
                {
                    "offset": point_offset,
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                    "tilt_x": tilt_x,
                    "tilt_y": tilt_y,
                    "pressure": pressure,
                    "time_delta": time_delta,
                    "raw_hex": data[point_offset : point_offset + POINT_SIZE].hex(" "),
                }
            )
        out_of_bounds = [
            point for point in points if not (0.0 <= point["x"] <= width and 0.0 <= point["y"] <= height)
        ]
        if out_of_bounds:
            failures.append(f"stroke {stroke_id} contains out-of-page coordinates")
        strokes.append(
            {
                "stroke_id": stroke_id,
                "stroke_offset": stroke_offset,
                "stroke_size": stroke_size,
                "point_count": len(points),
                "points": points,
                "bounds": _bounds(points),
                "shape_association": stroke_id in shape_set if shape_ids is not None else None,
            }
        )
    if shape_ids is not None:
        missing_shape_ids = sorted(set(entry["stroke_id"] for entry in entries) - shape_set)
        if missing_shape_ids:
            failures.append(f"{len(missing_shape_ids)} point index stroke IDs absent from shape field 1")
    total_points = sum(stroke["point_count"] for stroke in strokes)
    status = "matched" if not failures and len(strokes) > 0 else "partially_matches"
    result = _result(status, evidence, failures)
    result.update(
        {
            "schema": {
                "page_id": page_id,
                "point_resource_id": point_resource_id,
                "header_size": EXPECTED_HEADER_SIZE,
                "point_size": POINT_SIZE,
                "point_struct": ">ffBBHI",
                "index_entry_size": INDEX_ENTRY_SIZE,
                "byte_order": "big-endian",
                "index_start": index_start,
            },
            "source_sha256": _sha256(data),
            "entries": entries,
            "strokes": strokes,
            "point_count": total_points,
            "stroke_count": len(strokes),
            "coordinate_system": {
                "native_page_size": {"width": width, "height": height},
                "transform": "identity",
                "bounds_validated": not any("out-of-page" in failure for failure in failures),
            },
        }
    )
    return result


def _result(status: str, evidence: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    return {"status": status, "evidence": evidence, "failures": failures}


def render_probe_strokes(path: Path, size: tuple[int, int], strokes: list[dict[str, Any]]) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = [(round(point["x"]), round(point["y"])) for point in stroke["points"]]
        if len(points) > 1:
            draw.line(points, fill=(0, 0, 0), width=2, joint="curve")
    image.save(path, format="PNG", optimize=False)

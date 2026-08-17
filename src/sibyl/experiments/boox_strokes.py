"""Evidence-preserving inspection of native BOOX NOTE handwriting resources."""

# The resource records intentionally mirror the compact artifact schema.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from sibyl.corpus import read_boox_metadata
from sibyl.experiments.boox_forensics import write_page_forensics

UUID_RE = re.compile(
    rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
DEFAULT_PAGE = 4
DEFAULT_NOTE = Path("samples/Grafting 101.note")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resource(path: str, data: bytes, *, kind: str) -> dict[str, Any]:
    return {"path": path, "kind": kind, "size": len(data), "sha256": _sha256(data)}


def _uuids(data: bytes) -> list[str]:
    return [match.decode("ascii").lower() for match in UUID_RE.findall(data)]


def _decode_points(data: bytes, width: float, height: float) -> tuple[list[dict[str, float]], str]:
    """Decode only the observed testable format; reject mixed/implausible records."""
    if len(data) < 72 or (len(data) - 72) % 16:
        return [], "unsupported_length"
    import struct

    points: list[dict[str, float]] = []
    for offset in range(72, len(data), 16):
        x, y = struct.unpack(">ff", data[offset : offset + 8])
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            return [], "coordinate_encoding_uncertain"
        points.append({"x": round(float(x), 6), "y": round(float(y), 6)})
    return points, "big_endian_float_xy_stride_16"


def _bounds(points: list[dict[str, float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs, ys = [point["x"] for point in points], [point["y"] for point in points]
    return {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}


def _draw(path: Path, size: tuple[int, int], strokes: list[dict[str, Any]]) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = [(round(point["x"]), round(point["y"])) for point in stroke["mapped_points"]]
        if len(points) >= 2:
            draw.line(points, fill=(0, 0, 0), width=2, joint="curve")
        elif points:
            draw.ellipse(
                (points[0][0] - 1, points[0][1] - 1, points[0][0] + 1, points[0][1] + 1),
                fill=(0, 0, 0),
            )
    image.save(path, format="PNG", optimize=False)


def _pdf_page(pdf: Path, page: int, output: Path, size: tuple[int, int]) -> bool:
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-png",
                "-r",
                "72",
                str(pdf),
                str(output.with_suffix("")),
            ],
            check=True,
            capture_output=True,
        )
        rendered = output.with_suffix(".png")
        if rendered != output:
            rendered.replace(output)
        return Image.open(output).size == size
    except (OSError, subprocess.CalledProcessError):
        return False


def inspect_boox_strokes(
    note_path: Path, *, page: int = DEFAULT_PAGE, output: Path
) -> dict[str, Any]:
    if not note_path.is_file():
        raise FileNotFoundError(f"Source file not found: {note_path}")
    if page < 1:
        raise ValueError("page must be one-based")
    note = read_boox_metadata(note_path)
    if page > note["page_count"]:
        raise ValueError(f"page {page} is outside the note ({note['page_count']} pages)")
    selected = note["pages"][page - 1]
    page_id = selected["page_id"]
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / "raw"
    raw_output.mkdir(exist_ok=True)
    with zipfile.ZipFile(note_path) as archive:
        entries = [
            (name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")
        ]
    page_entries = [(name, data) for name, data in entries if page_id in name]
    shapes = [(name, data) for name, data in page_entries if "/shape/" in name]
    points = [
        (name, data)
        for name, data in page_entries
        if "/point/" in name and name.endswith("#points")
    ]
    point_ids = {Path(name).name.split("#")[1]: (name, data) for name, data in points}
    strokes: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_resources = []
    for name, data in page_entries:
        resource = _resource(
            name,
            data,
            kind="shape"
            if "/shape/" in name
            else "point"
            if "/point/" in name
            else "page_resource",
        )
        preserved_path = raw_output / Path(name).name
        preserved_path.write_bytes(data)
        resource["preserved_path"] = str(preserved_path)
        raw_resources.append(resource)
    forensics = write_page_forensics(note_path, page=page, output=output)
    for order, (shape_name, shape_data) in enumerate(shapes):
        shape_id = Path(shape_name).name.split("#")[1]
        referenced = [uid for uid in _uuids(shape_data) if uid in point_ids]
        candidate_ids: list[str | None] = [uid for uid in referenced]
        if not candidate_ids:
            candidate_ids.append(None)
        for point_id in candidate_ids:
            point_data = point_ids[point_id][1] if point_id else None
            decoded, encoding = (
                _decode_points(point_data, selected["width"], selected["height"])
                if point_data
                else ([], "no_point_resource")
            )
            if not decoded:
                warnings.append(f"point decoding incomplete for {point_id or shape_id}: {encoding}")
            strokes.append(
                {
                    "order": order,
                    "shape_id": shape_id,
                    "point_resource_id": point_id,
                    "point_count": len(decoded),
                    "native_points": decoded,
                    "mapped_points": decoded,
                    "native_bounds": _bounds(decoded),
                    "mapped_bounds": _bounds(decoded),
                    "pen": {"metadata": "not_confidently_decoded"},
                    "source_resource_sha256": _sha256(point_data)
                    if point_data
                    else _sha256(shape_data),
                    "coordinate_mapping": {
                        "type": "identity",
                        "confidence": "explicit_only_when_decoded",
                    },
                }
            )
    manifest_path = Path("samples/Grafting-101-corpus.json")
    mapping: dict[str, Any] = {
        "boox_page": page,
        "boox_page_id": page_id,
        "confidence": "unavailable",
    }
    pdf_path: Path | None = None
    if manifest_path.is_file():
        corpus = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = corpus.get("mapping", {}).get("note_page_ids", [])
        if page <= len(ids) and ids[page - 1] == page_id:
            pdf_path = Path(corpus["source_pdf"])
            mapping.update(
                {
                    "pdf_page": page,
                    "pdf_dimensions": corpus.get("page_size_points"),
                    "confidence": corpus.get("mapping", {}).get("confidence", "verified"),
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path.read_bytes()),
                }
            )
        else:
            warnings.append("corpus manifest does not confirm selected page ID")
    size = (round(selected["width"]), round(selected["height"]))
    native_path, strokes_path = (
        output / f"page-{page:03d}-native.png",
        output / f"page-{page:03d}-strokes.png",
    )
    _draw(native_path, size, strokes)
    _draw(strokes_path, size, strokes)
    pdf_image = output / f"page-{page:03d}-pdf.png"
    comparison: dict[str, Any] = {"pdf_rendered": False}
    if pdf_path and pdf_path.is_file() and _pdf_page(pdf_path, page, pdf_image, size):
        comparison["pdf_rendered"] = True
        overlay = Image.blend(Image.open(pdf_image).convert("RGB"), Image.open(native_path), 0.5)
        overlay.save(output / f"page-{page:03d}-overlay.png", format="PNG", optimize=False)
        ImageChops.difference(Image.open(pdf_image).convert("RGB"), Image.open(native_path)).save(
            output / f"page-{page:03d}-diff.png", format="PNG", optimize=False
        )
    metadata: dict[str, Any] = {
        "artifact": "boox_strokes",
        "note_format": "ZIP container with BOOX protobuf-like resources",
        "source_note": str(note_path),
        "source_note_sha256": _sha256(note_path.read_bytes()),
        "page_order": note["pages"],
        "selected_page": selected,
        "page_mapping": mapping,
        "shape_resource_ids": [Path(name).name.split("#")[1] for name, _data in shapes],
        "point_resource_ids": sorted(point_ids),
        "source_pdf": str(pdf_path) if pdf_path else None,
        "source_pdf_sha256": _sha256(pdf_path.read_bytes()) if pdf_path else None,
        "resources": raw_resources,
        "strokes": strokes,
        "raw_resource_preservation": True,
        "reconstruction": {
            "native_dimensions": list(size),
            "coordinate_transform": "identity",
            "stroke_width": 2,
            "background": "white",
        },
        "comparison": comparison,
        "warnings": sorted(set(warnings)),
        "forensics": {
            "artifact": str(output / f"page-{page:03d}-forensics.json"),
            "point_wire_candidate_count": forensics["wire_format"]["point_candidate_count"],
            "shape_wire_candidate_count": forensics["wire_format"]["shape_candidate_count"],
        },
    }
    metadata_path = output / f"page-{page:03d}-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_path = output.parent / "boox-strokes.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def format_boox_strokes(result: dict[str, Any]) -> str:
    return f"page: {result['selected_page']['note_page']}\nstrokes: {len(result['strokes'])}\noutput: {result['reconstruction']['native_dimensions']}"

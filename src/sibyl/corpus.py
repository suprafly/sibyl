"""Deterministic source-preserving corpus preparation utilities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

NOTE_PAGE_SIZE = (1404.0, 1872.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fragment(text: str, key: str) -> Any:
    marker = f'"{key}":'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"BOOX note metadata is missing {key}")
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(text[start + len(marker) :])
    return value


def read_boox_metadata(note_path: Path) -> dict[str, Any]:
    """Read only the BOOX page-order and dimension metadata needed for mapping."""
    try:
        with zipfile.ZipFile(note_path) as archive:
            candidates = [
                name for name in archive.namelist() if name.endswith("/note/pb/note_info")
            ]
            if len(candidates) != 1:
                raise ValueError("BOOX note must contain exactly one note_info metadata entry")
            text = archive.read(candidates[0]).decode("utf-8", errors="ignore")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Unable to read BOOX note metadata: {note_path}") from error
    page_names = _json_fragment(text, "pageNameList")
    page_info = _json_fragment(text, "pageInfoMap")
    if (
        not isinstance(page_names, list)
        or not page_names
        or not all(isinstance(item, str) for item in page_names)
    ):
        raise ValueError("BOOX note pageNameList is missing or invalid")
    if not isinstance(page_info, dict):
        raise ValueError("BOOX note pageInfoMap is missing or invalid")
    pages: list[dict[str, Any]] = []
    for number, page_id in enumerate(page_names, start=1):
        info = page_info.get(page_id)
        if not isinstance(info, dict):
            raise ValueError(f"BOOX note is missing metadata for page {page_id}")
        width, height = info.get("width"), info.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise ValueError(f"BOOX note has invalid dimensions for page {page_id}")
        pages.append(
            {
                "note_page": number,
                "page_id": page_id,
                "width": float(width),
                "height": float(height),
            }
        )
    return {"page_count": len(pages), "pages": pages, "metadata_entry": candidates[0]}


def _pdf_info(pdf_path: Path) -> tuple[int, tuple[float, float]]:
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Unable to inspect PDF metadata: {pdf_path}") from error
    pages_match = re.search(r"^Pages:\s+(\d+)$", result.stdout, re.MULTILINE)
    size_match = re.search(r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts$", result.stdout, re.MULTILINE)
    if pages_match is None or size_match is None:
        raise ValueError(f"PDF metadata lacks page count or page size: {pdf_path}")
    return int(pages_match.group(1)), (float(size_match.group(1)), float(size_match.group(2)))


def _validate_mapping(
    note: dict[str, Any], pdf_count: int, pdf_size: tuple[float, float]
) -> list[int]:
    if note["page_count"] != pdf_count:
        raise ValueError(
            "cannot establish BOOX/PDF mapping: "
            f"note has {note['page_count']} pages, PDF has {pdf_count}"
        )
    note_sizes = {(page["width"], page["height"]) for page in note["pages"]}
    if len(note_sizes) != 1 or next(iter(note_sizes)) != pdf_size:
        raise ValueError(
            "cannot establish BOOX/PDF mapping: page dimensions do not match "
            f"(note={sorted(note_sizes)}, pdf={pdf_size})"
        )
    return list(range(1, pdf_count + 1))


def _copy_pages(source_pdf: Path, output_pdf: Path, pages: list[int]) -> None:
    if shutil.which("pdfseparate") is None or shutil.which("pdfunite") is None:
        raise RuntimeError("corpus preparation requires Poppler pdfseparate and pdfunite")
    with tempfile.TemporaryDirectory(prefix="sibyl-corpus-") as directory:
        separated = Path(directory) / "page-%d.pdf"
        subprocess.run(["pdfseparate", str(source_pdf), str(separated)], check=True)
        page_paths = [Path(directory) / f"page-{page}.pdf" for page in pages]
        missing = [str(path) for path in page_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"PDF page extraction did not produce: {', '.join(missing)}")
        subprocess.run(
            ["pdfunite", *(str(path) for path in page_paths), str(output_pdf)], check=True
        )


def prepare_boox_corpus(
    note_path: Path,
    source_pdf: Path,
    *,
    output_pdf: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    for path in (note_path, source_pdf):
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")
    if output_pdf.resolve() in {note_path.resolve(), source_pdf.resolve()}:
        raise ValueError("output PDF must not overwrite a source file")
    note = read_boox_metadata(note_path)
    pdf_count, pdf_size = _pdf_info(source_pdf)
    selected_pages = _validate_mapping(note, pdf_count, pdf_size)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    _copy_pages(source_pdf, output_pdf, selected_pages)
    output_count, output_size = _pdf_info(output_pdf)
    if output_count != len(selected_pages) or output_size != pdf_size:
        raise RuntimeError("reduced PDF verification failed")
    manifest = {
        "artifact": "boox_corpus_pdf",
        "source_pdf": str(source_pdf),
        "source_pdf_sha256": _sha256(source_pdf),
        "source_note": str(note_path),
        "source_note_sha256": _sha256(note_path),
        "selected_source_pages": selected_pages,
        "output_pdf": str(output_pdf),
        "manifest_path": str(manifest_path),
        "output_pdf_sha256": _sha256(output_pdf),
        "page_count": output_count,
        "page_size_points": {"width": output_size[0], "height": output_size[1]},
        "mapping": {
            "type": "identity_by_order",
            "confidence": "verified",
            "assumptions": [
                "The BOOX note pageNameList order corresponds to PDF page order.",
                "The note contains the same page count as the PDF.",
                "Every BOOX page dimension matches the PDF page size in points.",
                "Pages were copied with Poppler pdfseparate/pdfunite; "
                "no raster rendering was used.",
            ],
            "note_page_ids": [page["page_id"] for page in note["pages"]],
            "note_metadata_entry": note["metadata_entry"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def format_corpus_result(manifest: dict[str, Any]) -> str:
    return (
        f"output: {manifest['output_pdf']}\n"
        f"pages: {manifest['page_count']}\n"
        f"manifest: {manifest['manifest_path']}"
    )

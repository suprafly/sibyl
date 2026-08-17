import hashlib
import json
from pathlib import Path

import pytest

from sibyl.corpus import prepare_boox_corpus, read_boox_metadata

NOTE = Path("samples/Grafting 101.note")
PDF = Path("samples/Grafting 101.pdf")


def test_boox_metadata_has_ordered_pages_and_dimensions() -> None:
    metadata = read_boox_metadata(NOTE)
    assert metadata["page_count"] == 13
    assert [page["note_page"] for page in metadata["pages"]] == list(range(1, 14))
    assert {(page["width"], page["height"]) for page in metadata["pages"]} == {(1404.0, 1872.0)}


def test_reduced_pdf_preserves_source_and_records_provenance(tmp_path: Path) -> None:
    output = tmp_path / "corpus.pdf"
    manifest_path = tmp_path / "corpus.json"
    source_hash = hashlib.sha256(PDF.read_bytes()).hexdigest()
    note_hash = hashlib.sha256(NOTE.read_bytes()).hexdigest()
    manifest = prepare_boox_corpus(NOTE, PDF, output_pdf=output, manifest_path=manifest_path)
    assert manifest["selected_source_pages"] == list(range(1, 14))
    assert manifest["page_count"] == 13
    assert manifest["source_pdf_sha256"] == source_hash
    assert manifest["source_note_sha256"] == note_hash
    assert manifest["output_pdf_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert NOTE.read_bytes().startswith(b"PK\x03\x04")
    assert PDF.read_bytes()[:5] == b"%PDF-"


def test_mapping_mismatch_refuses_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sibyl.corpus._pdf_info", lambda _path: (12, (1404.0, 1872.0)))
    with pytest.raises(ValueError, match="note has 13 pages, PDF has 12"):
        prepare_boox_corpus(
            NOTE,
            PDF,
            output_pdf=tmp_path / "corpus.pdf",
            manifest_path=tmp_path / "corpus.json",
        )
    assert not (tmp_path / "corpus.pdf").exists()


def test_output_cannot_overwrite_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_boox_corpus(
            NOTE,
            PDF,
            output_pdf=PDF,
            manifest_path=tmp_path / "corpus.json",
        )

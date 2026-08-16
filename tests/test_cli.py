import json
from pathlib import Path

import pytest
from PIL import Image

from sibyl.cli import main
from sibyl.transform import RegionBounds, TransformedPage, TransformedRegion


def test_cli_accepts_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_run_requires_an_image(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run"])
    assert error.value.code == 2
    assert "required: image" in capsys.readouterr().err


def test_legacy_recover_command_is_not_exposed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["recover", "page.png"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def _projected_page(tmp_path: Path) -> TransformedPage:
    source = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(source)
    crop = tmp_path / "assets" / "figure-01.png"
    crop.parent.mkdir()
    Image.new("RGB", (4, 4), "white").save(crop)
    bounds = RegionBounds(0, 0, 10, 10)
    return TransformedPage(
        source={"image": str(source)},
        dimensions={"width": 20, "height": 20},
        interpretation={},
        regions=[
            TransformedRegion(
                order=0,
                kind="heading",
                bounds=bounds,
                prepared_bounds=bounds,
                qwen_text="Heading",
                text="Unusual Speling!",
                source={"image": str(source), "bounds": {"left": 0}, "crop": str(crop)},
                recognizer={},
            ),
            TransformedRegion(
                order=1,
                kind="figure",
                bounds=bounds,
                prepared_bounds=bounds,
                qwen_text="diagram",
                text="",
                source={"image": str(source), "bounds": {"left": 0}, "crop": str(crop)},
                recognizer={},
            ),
        ],
        runtime={
            "benchmark": {"region_count": 2},
            "disagreements": [{"order": 0, "qwen": "Heading", "trocr": "Unusual Speling!"}],
        },
    )


def test_run_defaults_to_text_and_preserves_model_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sibyl.cli.transform_page", lambda image: _projected_page(tmp_path))
    assert main(["run", str(tmp_path / "page.png")]) == 0
    output = capsys.readouterr().out
    assert output.strip() == "Unusual Speling!"
    assert "diagram" not in output


def test_run_json_is_the_complete_structured_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sibyl.cli.transform_page", lambda image: _projected_page(tmp_path))
    assert main(["run", str(tmp_path / "page.png"), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"regions"' in output
    assert '"benchmark"' in output


def test_run_markdown_writes_projection_and_asset_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sibyl.cli.transform_page", lambda image: _projected_page(tmp_path))
    assert main(["run", str(tmp_path / "page.png"), "--markdown"]) == 0
    output = capsys.readouterr().out
    markdown = tmp_path / "page.sibyl" / "transform.md"
    assert str(markdown.parent) in output
    assert "![Figure 1](assets/figure-01.png)" in markdown.read_text()
    transform = tmp_path / "page.sibyl" / "transform.json"
    structured = json.loads(transform.read_text())
    assert structured["interpretation"] == {}
    assert structured["regions"][0]["qwen_text"] == "Heading"
    assert structured["regions"][0]["text"] == "Unusual Speling!"
    assert structured["runtime"]["disagreements"][0]["qwen"] == "Heading"
    assert (tmp_path / "page.sibyl" / "assets" / "figure-01.png").exists()


def test_run_output_projections_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "page.png", "--markdown", "--json"])
    assert error.value.code == 2

from pathlib import Path

import pytest
from PIL import Image

from sibyl.cli import main
from sibyl.experiments.trocr import MODEL_ID, ExperimentResult, format_result, run_experiment


class FakeRecognizer:
    def recognize(self, image: Image.Image) -> tuple[str, float]:
        assert image.mode == "RGB"
        return "hello from a test", 12.5


def test_trocr_requires_an_image(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["experiment", "trocr"])
    assert error.value.code == 2
    assert "required: image" in capsys.readouterr().err


def test_missing_image_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    assert main(["experiment", "trocr", str(missing)]) == 2
    assert "Image not found" in capsys.readouterr().err


def test_unsupported_image_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    image_path = tmp_path / "not-an-image.bin"
    image_path.write_text("not an image")
    assert main(["experiment", "trocr", str(image_path)]) == 2
    assert "Unable to read image" in capsys.readouterr().err


def test_provider_boundary_and_json_output(tmp_path: Path) -> None:
    image_path = tmp_path / "line.png"
    Image.new("L", (20, 10), color=255).save(image_path)
    result = run_experiment(
        image_path,
        FakeRecognizer(),
        model_load_ms=3.0,
        cuda_available=False,
        device="cpu",
    )
    assert result.model == MODEL_ID
    assert result.text == "hello from a test"
    assert result.device == "cpu"
    assert result.gpu is None
    assert '"text": "hello from a test"' in format_result(result, as_json=True)


def test_result_structure() -> None:
    fields = set(ExperimentResult.__dataclass_fields__)
    assert {
        "source_image",
        "model",
        "device",
        "cuda_available",
        "gpu",
        "inference_ms",
        "text",
    } <= fields

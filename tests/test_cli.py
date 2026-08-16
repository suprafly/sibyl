import pytest

from sibyl.cli import main


def test_cli_accepts_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_recover_requires_an_image(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["recover"])
    assert error.value.code == 2
    assert "required: image" in capsys.readouterr().err

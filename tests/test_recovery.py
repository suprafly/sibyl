import json
from pathlib import Path
from typing import Any

from PIL import Image

from sibyl.recovery import format_recovery, recover_page


class FakeInterpreter:
    model = "fake-qwen"

    def interpret(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        assert image.mode == "L"
        assert image.size == (100, 100)
        return {
            "page_interpretation": {"layout": "heading and note"},
            "regions": [
                {
                    "order": 1,
                    "kind": "heading",
                    "left": 0.1,
                    "top": 0.1,
                    "right": 0.9,
                    "bottom": 0.4,
                    "text": "Calamodin",
                }
            ],
        }, 4.0

    def release(self) -> None:
        pass


class FakeRecognizer:
    def recognize(self, image: Image.Image) -> tuple[str, float]:
        assert image.mode == "RGB"
        assert image.size == (80, 30)
        return "Calamondin", 2.5


def test_recovery_preserves_source_qwen_text_and_recognizer_evidence(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    page = recover_page(
        image_path,
        FakeInterpreter(),
        FakeRecognizer(),
        recognizer_metadata={"device": "cpu"},
    )
    assert page.dimensions == {"width": 100, "height": 100}
    assert page.regions[0].qwen_text == "Calamodin"
    assert page.regions[0].text == "Calamondin"
    assert page.regions[0].bounds.left == 10
    assert json.loads(format_recovery(page))["regions"][0]["source"]["image"] == str(image_path)

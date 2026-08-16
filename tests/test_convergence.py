import json
from pathlib import Path

import pytest

from sibyl.experiments.convergence import normalize_reading, run_convergence


def artifact(tmp_path: Path, *, qwen: list[str], trocr: list[str], figure: bool = False) -> Path:
    regions = []
    for number, (q, t) in enumerate(zip(qwen, trocr, strict=False), start=1):
        regions.append(
            {
                "region_id": f"region-{number:02d}",
                "crop": {
                    "path": f"crop-{number}.png",
                    "sha256": f"hash-{number}",
                    "source_bbox": {"top": number},
                },
                "qwen": {"runs": [{"run": i, "status": "ok", "text": q} for i in range(1, 3)]},
                "trocr": {"runs": [{"run": i, "status": "ok", "text": t} for i in range(1, 3)]},
            }
        )
    path = tmp_path / "trocr-compare.json"
    path.write_text(
        json.dumps({"experiment": "trocr_compare", "source": "missing.png", "regions": regions}),
        encoding="utf-8",
    )
    return path


def test_normalization_is_superficial_only() -> None:
    assert normalize_reading("  transports   water . ") == "transports water"
    assert normalize_reading("transpirs") == "transpirs"


def test_cross_model_agreement_and_provenance(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["Xylem"], trocr=["Xylem ."])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    region = result["regions"][0]
    assert region["candidate"] == "Xylem"
    assert "cross_model_agreement" in region["basis"]
    assert region["source_crop"]["sha256"] == "hash-1"
    assert len(region["observations"]["qwen"]) == 2


def test_disagreement_is_not_majority_voting(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["stable wrong"], trocr=["different reading"])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert result["regions"][0]["candidate"] == "[unclear]"
    assert result["regions"][0]["human_confirmed"] is False


def test_human_review_overrides_ambiguity_and_is_traceable(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["uncertain"], trocr=["other"])
    review = tmp_path / "review.yaml"
    review.write_text(
        'regions:\n  region-01:\n    text: "Confirmed reading"\n    confirmed: true\n',
        encoding="utf-8",
    )
    result = run_convergence(
        input_path,
        review_path=review,
        markdown_path=tmp_path / "out.md",
        json_path=tmp_path / "out.json",
    )
    assert result["regions"][0]["candidate"] == "Confirmed reading"
    assert result["regions"][0]["basis"] == ["human_review"]
    assert result["review_input"] == str(review)


def test_missing_malformed_and_malformed_review_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_convergence(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        run_convergence(bad)
    input_path = artifact(tmp_path, qwen=["a"], trocr=["b"])
    review = tmp_path / "bad.yaml"
    review.write_text("regions:\n  region-01:\n    text: nope\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_convergence(input_path, review_path=review)


def test_figure_reference_is_preserved_from_canonical_observation(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"not an image")
    canonical = tmp_path / "page.sibyl"
    canonical.mkdir()
    (canonical / "transform.json").write_text(
        json.dumps(
            {
                "page_text": [],
                "regions": [{"kind": "figure", "source": {"crop": "assets/figure-01.png"}}],
            }
        ),
        encoding="utf-8",
    )
    input_path = artifact(tmp_path, qwen=["heading"], trocr=["heading"])
    payload = json.loads(input_path.read_text())
    payload["source"] = str(source)
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert "![Figure 1](assets/figure-01.png)" in (tmp_path / "out.md").read_text()

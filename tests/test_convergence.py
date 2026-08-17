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
    assert "cross_model_overlap" in region["basis"]
    assert region["source_crop"]["sha256"] == "hash-1"
    assert len(region["observations"]["qwen"]) == 2


def test_disagreement_is_not_majority_voting(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["stable wrong"], trocr=["different reading"])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert result["regions"][0]["candidate"] == "[unclear]"
    assert result["regions"][0]["human_confirmed"] is False


def test_variable_qwen_and_stable_trocr_use_partial_phrase(tmp_path: Path) -> None:
    input_path = artifact(
        tmp_path,
        qwen=["Splic grafting - what", "Splia grafting - what", "Splea grafting - what"],
        trocr=["splice grafting - what", "splice grafting - what", "splice grafting - what"],
    )
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    region = result["regions"][0]
    assert region["candidate"] == "splice grafting - what"
    assert "cross_model_overlap" in region["basis"]
    assert region["evidence"]["cross_model_overlap"]
    assert region["evidence"]["common_phrases"][0]["token_count"] >= 2


def test_partial_sentence_constructs_candidate_from_token_support(tmp_path: Path) -> None:
    input_path = artifact(
        tmp_path,
        qwen=[
            "- transports mineral nutrients on water from root to",
            "- transpirs mineral nutrients on water from root to",
            "- transports mineral nutrients on water from root to",
        ],
        trocr=["- transports interval not trans"] * 3,
    )
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    candidate = result["regions"][0]["candidate"]
    assert candidate.startswith("- transports mineral nutrients")
    assert candidate != "[unclear]"


def test_incompatible_readings_remain_unclear_and_output_is_deterministic(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["alpha beta"], trocr=["gamma delta"])
    first = run_convergence(
        input_path, markdown_path=tmp_path / "one.md", json_path=tmp_path / "one.json"
    )
    second = run_convergence(
        input_path, markdown_path=tmp_path / "two.md", json_path=tmp_path / "two.json"
    )
    assert first["regions"][0]["candidate"] == "[unclear]"
    assert first["regions"] == second["regions"]
    assert "alpha" not in Path("src/sibyl/experiments/convergence.py").read_text()


def test_document_pass_preserves_regional_layer_and_exposes_scoring(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["transpirs food"], trocr=["transports food"])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    regional = result["regions"][0]
    document = result["document_convergence"]["regions"][0]
    assert regional["candidate"]
    assert document["regional_candidate"] == regional["candidate"]
    assert "recognition_support" in document["basis"]
    assert result["document_candidate"]["regions"] == [document["selected"]]


def test_document_pass_orders_spatially_and_joins_same_line_regions(tmp_path: Path) -> None:
    input_path = artifact(
        tmp_path, qwen=["we will do now", "what"], trocr=["we will do now", "what"]
    )
    payload = json.loads(input_path.read_text())
    payload["regions"][0]["crop"]["source_bbox"] = {
        "left": 200,
        "top": 10,
        "right": 400,
        "bottom": 100,
    }
    payload["regions"][1]["crop"]["source_bbox"] = {
        "left": 10,
        "top": 10,
        "right": 190,
        "bottom": 100,
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert result["document_convergence"]["regions"][0]["region_id"] == "region-02"
    assert result["document_convergence"]["blocks"] == ["what we will do now"]


def test_unrelated_regions_are_not_joined(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["heading", "paragraph"], trocr=["heading", "paragraph"])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert len(result["document_convergence"]["blocks"]) == 2


def test_document_review_is_authoritative_but_model_evidence_remains(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["uncertain"], trocr=["other"])
    review = tmp_path / "review.yaml"
    review.write_text(
        'regions:\n  region-01:\n    text: "Human reading"\n    confirmed: true\n',
        encoding="utf-8",
    )
    result = run_convergence(
        input_path,
        review_path=review,
        markdown_path=tmp_path / "out.md",
        json_path=tmp_path / "out.json",
    )
    decision = result["document_convergence"]["regions"][0]
    assert decision["selected"] == "Human reading"
    assert decision["human_confirmed"] is True
    assert result["regions"][0]["observations"]["qwen"][0]["text"] == "uncertain"


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
    run_convergence(input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json")
    assert "![Figure 1](assets/figure-01.png)" in (tmp_path / "out.md").read_text()


def _figure_source(tmp_path: Path, input_path: Path) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"not an image")
    canonical = tmp_path / "page.sibyl"
    canonical.mkdir()
    (canonical / "transform.json").write_text(
        json.dumps(
            {
                "page_text": [],
                "regions": [
                    {
                        "kind": "figure",
                        "bounds": {"left": 100, "top": 100, "right": 300, "bottom": 300},
                        "source": {"crop": "assets/figure-01.png"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(input_path.read_text())
    payload["source"] = str(source)
    payload["regions"][0]["crop"]["source_bbox"] = {
        "left": 120,
        "top": 120,
        "right": 280,
        "bottom": 280,
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")


def test_existing_figure_geometry_suppresses_duplicate_uncertainty_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    input_path = artifact(tmp_path, qwen=["diagram material"], trocr=["#"])
    _figure_source(tmp_path, input_path)
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    classification = result["regions"][0]["classification"]
    assert classification == {
        "kind": "figure",
        "basis": ["overlaps_existing_figure", "drawing_region_evidence"],
        "emitted": False,
        "represented_by": "Figure 1",
    }
    markdown = (tmp_path / "out.md").read_text()
    assert markdown.count("![Figure 1](assets/figure-01.png)") == 1
    assert "[unclear]" not in markdown
    assert result["regions"][0]["observations"]["qwen"]
    assert result["document_convergence"]["regions"][0]["emitted"] is False


def test_uncertain_text_without_figure_evidence_remains_unclear(tmp_path: Path) -> None:
    input_path = artifact(tmp_path, qwen=["alpha beta"], trocr=["gamma delta"])
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert result["regions"][0]["classification"]["kind"] == "unknown"
    assert "[unclear]" in (tmp_path / "out.md").read_text()


def test_observed_capitalization_is_projected_without_resolving_handwriting(
    tmp_path: Path,
) -> None:
    input_path = artifact(
        tmp_path,
        qwen=["Splic grafting - what"] * 3,
        trocr=["splice grafting - what"] * 3,
    )
    result = run_convergence(
        input_path, markdown_path=tmp_path / "out.md", json_path=tmp_path / "out.json"
    )
    assert "Splice grafting" in (tmp_path / "out.md").read_text()
    assert result["regions"][0]["candidate"] == "splice grafting - what"

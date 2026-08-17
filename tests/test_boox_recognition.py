import json
from pathlib import Path
from typing import Any

from PIL import Image

from sibyl.experiments import boox_recognition as experiment
from sibyl.experiments import handwriting_exemplars
from sibyl.experiments.qwen_recognition_knobs import extract_recognition_text


def _stroke(identifier: str, order: int, x: float, y: float) -> dict[str, Any]:
    points = [
        {"x": x, "y": y},
        {"x": x + 2, "y": y + 2},
    ]
    return {
        "stroke_id": identifier,
        "order": order,
        "point_count": len(points),
        "native_points": points,
        "native_bounds": {"left": x, "top": y, "right": x + 2, "bottom": y + 2},
        "shape_association": True,
    }


def test_native_selection_excludes_target_and_preserves_decoder_order() -> None:
    strokes = [_stroke("target", 1, 10, 10), _stroke("other", 0, 30, 30)]
    selected = experiment.select_native_strokes(
        strokes, (0, 0, 20, 20), exclude_ids={"target"}
    )
    assert selected == []
    assert [item["stroke_id"] for item in experiment.select_native_strokes(strokes)] == [
        "other",
        "target",
    ]


def test_native_rendering_and_evaluation_are_deterministic(tmp_path: Path) -> None:
    strokes = [_stroke("stroke-1", 0, 5, 6)]
    first = experiment.render_native_reference(tmp_path / "one.png", strokes)
    second = experiment.render_native_reference(tmp_path / "two.png", strokes)
    assert first["sha256"] == second["sha256"]
    assert first["stroke_ids"] == ["stroke-1"]
    assert experiment.evaluate_reading("Xylem?", "Xylem") == {
        "exact_match": False,
        "normalized_exact_match": False,
        "token_overlap": 1.0,
        "character_edit_distance": 1,
        "raw_exact_match": False,
        "word_overlap": 1.0,
        "unresolved_tokens": [],
        "reading": "Xylem?",
        "ground_truth": "Xylem",
    }


def test_matched_native_rendering_records_presentation_parameters(tmp_path: Path) -> None:
    record = experiment.render_native_reference(
        tmp_path / "matched.png",
        [_stroke("stroke-1", 0, 5, 6)],
        stroke_width=6,
        presentation_height=32,
    )
    assert record["dimensions"]["height"] == 32
    assert record["native_dimensions"]["height"] == 1872
    assert record["rendering"] == {
        "background": "white",
        "coordinate_transform": "identity",
        "crop": "full_page",
        "presentation_height": 32,
        "presentation_scale": 32 / 1872,
        "resampling": "lanczos",
        "stroke_width": 6,
    }


def test_condition_selection_is_canonical_and_rejects_unknown_values() -> None:
    assert experiment.selected_conditions("leave-one-region-out,baseline") == (
        "baseline",
        "leave-one-region-out",
    )
    try:
        experiment.selected_conditions("baseline,baseline")
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate conditions should be rejected")


def test_reference_line_selection_is_canonical_and_validated() -> None:
    catalog = [
        {"reference_id": "line-b"},
        {"reference_id": "line-a"},
    ]
    assert [item["reference_id"] for item in experiment._reference_lines(catalog, "line-a")] == [
        "line-a"
    ]
    try:
        experiment._reference_lines(catalog, "missing")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("unknown reference lines should be rejected")


def test_qwen_response_extraction_prefers_content_and_falls_back_to_thinking() -> None:
    assert (
        extract_recognition_text(
            {"message": {"content": "on the water from root to"}}
        )
        == "on the water from root to"
    )
    assert (
        extract_recognition_text(
            {"message": {"thinking": "on the water from root to"}}
        )
        == "on the water from root to"
    )
    assert (
        extract_recognition_text(
            {
                "message": {
                    "content": "returned transcription",
                    "thinking": "reasoning containing another transcription",
                }
            }
        )
        == "returned transcription"
    )
    assert (
        extract_recognition_text(
            {"message": {"content": "", "thinking": "usable thinking transcription"}}
        )
        == "usable thinking transcription"
    )
    assert extract_recognition_text({"message": {"content": "", "thinking": ""}}) is None


def test_recovery_selection_requires_agreement_and_preserves_candidates() -> None:
    unresolved = experiment._select_recovery_reading(
        {"readings": ["さくいお\uFF5E", "sclo~", "scio~"]}
    )
    assert unresolved == {
        "status": "unresolved",
        "reading": None,
        "rule": "no_majority",
        "reason": "successful_runs_disagree",
        "candidates": ["さくいお\uFF5E", "sclo~", "scio~"],
    }
    selected = experiment._select_recovery_reading(
        {"readings": ["sclio~", "scion", "scion"]}
    )
    assert selected["status"] == "selected"
    assert selected["reading"] == "scion"
    assert selected["rule"] == "normalized_majority"
    assert selected["candidates"] == ["sclio~", "scion", "scion"]


def test_truncated_empty_content_preserves_thinking_as_unconfirmed_evidence(
    monkeypatch: Any,
) -> None:
    raw = {
        "done_reason": "length",
        "eval_count": 256,
        "message": {"content": "", "thinking": "on water from root to"},
    }
    monkeypatch.setattr(handwriting_exemplars, "_query", lambda **_: (raw, 1.0))
    reader = handwriting_exemplars.OllamaExemplarReader(model="qwen-test")
    result, _ = reader.read([Image.new("RGB", (4, 4), "white")], "prompt", {"num_predict": 1024})
    assert result["status"] == "truncated_response"
    assert result["text"] == "on water from root to"
    analysis = handwriting_exemplars._read_configuration(
        reader,
        [Image.new("RGB", (4, 4), "white")],
        "prompt",
        {"num_predict": 1024},
        1,
    )
    assert analysis["readings"] == []
    assert analysis["runs"][0]["status"] == "truncated_response"
    boox_analysis = experiment._condition_analysis(analysis)
    experiment._add_truncated_evidence(boox_analysis)
    assert boox_analysis["truncated_evidence"] == ["on water from root to"]
    assert boox_analysis["stable_reading"] is None


class FakeReader:
    model = "qwen-test"

    def __init__(self, observer: Any) -> None:
        self.observer = observer
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def read(
        self, images: list[Image.Image], prompt: str, controls: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        raw = {
            "message": {
                "content": '{"text": "candidate"}',
                "thinking": "reasoning preserved separately",
            }
        }
        self.observer(raw)
        return {"status": "ok", "text": "candidate", "raw_response": raw}, 0.0

    def release(self) -> None:
        return None


def test_run_preserves_conditions_provenance_and_nonleaking_review(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1404, 1872), "white").save(image_path)
    note_path = tmp_path / "page.note"
    note_path.write_bytes(b"note")
    target_path = tmp_path / "target.png"
    Image.new("RGB", (20, 20), "white").save(target_path)
    strokes = [_stroke("target-stroke", 0, 15, 15), _stroke("reference-stroke", 1, 70, 70)]
    monkeypatch.setattr(
        experiment,
        "_targets",
        lambda *args, **kwargs: [
            {
                "target_id": "line-target",
                "kind": "line",
                "path": target_path,
                "source_bbox": {"left": 10, "top": 10, "right": 30, "bottom": 30},
            }
        ],
    )
    monkeypatch.setattr(
        experiment,
        "_line_catalog",
        lambda *args, **kwargs: [
            {
                "reference_id": "line-target",
                "source_bbox": {"left": 10, "top": 10, "right": 30, "bottom": 30},
                "bbox": (10, 10, 30, 30),
                "source_artifact": "reread.json",
            },
            {
                "reference_id": "line-reference",
                "source_bbox": {"left": 60, "top": 60, "right": 90, "bottom": 90},
                "bbox": (60, 60, 90, 90),
                "source_artifact": "reread.json",
            },
        ],
    )
    monkeypatch.setattr(
        experiment,
        "_verified_page",
        lambda _note: {
            "selected_page": {"note_page": 4},
            "reconstruction": {"native_dimensions": [1404, 1872]},
            "strokes": strokes,
        },
    )
    readers: list[FakeReader] = []

    def factory(observer: Any) -> FakeReader:
        reader = FakeReader(observer)
        readers.append(reader)
        return reader

    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_id": "line-target",
                        "transcription": "answer",
                        "confirmed": True,
                    }
                ]
            }
        )
    )
    artifact = experiment.run_boox_recognition(
        image_path,
        note_path=note_path,
        runs=2,
        review_path=review,
        output_path=tmp_path / "artifact.json",
        reader_factory=factory,
    )
    assert [result["condition"] for result in artifact["results"]] == [
        "baseline",
        "native-render",
        "native-exemplar",
        "multi-exemplar",
        "leave-one-region-out",
    ]
    assert artifact["results"][0]["image_order"] == ["line-target"]
    assert artifact["request_controls"]["num_predict"] == 1024
    assert artifact["request_controls"]["num_ctx"] == 8192
    assert artifact["reference_render_controls"] == {
        "native_stroke_width": 2,
        "reference_height": None,
        "reference_lines": ["line-target", "line-reference"],
    }
    assert artifact["results"][2]["reference_stroke_ids"] == ["reference-stroke"]
    assert "answer" not in artifact["results"][2]["prompt"]
    assert artifact["results"][2]["analysis"]["evaluation"]["ground_truth"] == "answer"
    assert (
        artifact["results"][0]["analysis"]["parsed_responses"][0]["thinking"]
        == "reasoning preserved separately"
    )
    assert (
        artifact["results"][0]["analysis"]["runs"][0]["raw_response"]["message"]["thinking"]
        == "reasoning preserved separately"
    )
    assert all(result["raw_response_observed"] for result in artifact["results"])
    assert artifact["status"] == "complete"
    assert len(artifact["completed_results"]) == 5
    assert json.loads((tmp_path / "artifact.json").read_text()) == artifact


def test_markdown_recovery_enumerates_lines_and_writes_auditable_projection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1404, 1872), "white").save(image_path)
    note_path = tmp_path / "page.note"
    note_path.write_bytes(b"note")
    target_paths = {
        identifier: tmp_path / f"{identifier}.png"
        for identifier in (
            "region-02-line-01",
            "region-02-line-02",
            "region-03-line-01",
        )
    }
    for path in target_paths.values():
        Image.new("RGB", (20, 20), "white").save(path)
    boxes: dict[str, dict[str, int]] = {
        identifier: {
            "left": 10 + index * 30,
            "top": 10,
            "right": 30 + index * 30,
            "bottom": 30,
        }
        for index, identifier in enumerate(target_paths)
    }
    targets: list[dict[str, Any]] = [
        {
            "target_id": identifier,
            "kind": "line",
            "path": path,
            "source_bbox": boxes[identifier],
        }
        for identifier, path in target_paths.items()
    ]
    catalog: list[dict[str, Any]] = [
        {
            "reference_id": identifier,
            "source_bbox": boxes[identifier],
            "bbox": (
                boxes[identifier]["left"],
                boxes[identifier]["top"],
                boxes[identifier]["right"],
                boxes[identifier]["bottom"],
            ),
            "source_artifact": "reread.json",
                "region_id": "region-02" if identifier.startswith("region-02") else "region-03",
        }
        for identifier in target_paths
    ]
    monkeypatch.setattr(experiment, "_targets", lambda *args, **kwargs: targets)
    monkeypatch.setattr(experiment, "_line_catalog", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(
        experiment,
        "_verified_page",
        lambda _note: {
            "selected_page": {"note_page": 4},
            "reconstruction": {"native_dimensions": [1404, 1872]},
            "strokes": [
                _stroke("stroke-1", 0, 15, 15),
                _stroke("stroke-2", 1, 45, 15),
                _stroke("stroke-3", 2, 75, 15),
            ],
        },
    )

    def factory(observer: Any) -> FakeReader:
        return FakeReader(observer)

    artifact = experiment.run_boox_recognition(
        image_path,
        note_path=note_path,
        runs=2,
        output_path=tmp_path / "experiment.json",
        reader_factory=factory,
        markdown=True,
        native_stroke_width=None,
    )
    assert artifact["recovery_requested"] is True
    assert artifact["request_controls"]["num_predict"] == 2048
    assert artifact["reference_render_controls"] == {
        "native_stroke_width": 6,
        "reference_height": 64,
            "reference_lines": [
                "region-02-line-01",
                "region-02-line-02",
                "region-03-line-01",
            ],
    }
    results = artifact["results"]
    assert all(result["condition"] == "leave-one-region-out" for result in results)
    assert [result["reference_stroke_ids"] for result in results] == [
        ["stroke-2"],
        ["stroke-1"],
        [],
    ]
    assert results[2]["reference_selection"]["reason"] == "insufficient_same_region_references"
    recovery_path = tmp_path / "page.sibyl" / "recovery.md"
    evidence_path = tmp_path / "page.sibyl" / "recovery.json"
    assert recovery_path.read_text() == "candidate\n\ncandidate\n\ncandidate\n"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["recovery"]["selection_condition"] == "leave-one-region-out"
    assert len(evidence["results"]) == 3

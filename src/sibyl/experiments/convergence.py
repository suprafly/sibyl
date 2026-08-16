"""Deterministic synthesis of preserved recognition evidence (experimental)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MARKDOWN = Path(".sibyl/experiments/converged.md")
DEFAULT_JSON = Path(".sibyl/experiments/convergence.json")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed convergence input: {path}") from error
    if not isinstance(value, dict) or value.get("experiment") != "trocr_compare":
        raise ValueError("input must be a trocr_compare artifact")
    if not isinstance(value.get("regions"), list):
        raise ValueError("input artifact is missing regions")
    return value


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].replace('\\"', '"')
    return value


def _load_review(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Review file not found: {path}")
    # The review contract is intentionally tiny; this parser avoids adding a runtime
    # dependency for two-level YAML mappings and accepts JSON as a useful subset.
    try:
        raw = path.read_text(encoding="utf-8")
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            regions = decoded.get("regions", {})
            if isinstance(regions, dict):
                decoded_result = {
                    str(key): dict(value)
                    for key, value in regions.items()
                    if isinstance(value, dict)
                }
                for region_id, review in decoded_result.items():
                    if not isinstance(review.get("text"), str) or not isinstance(
                        review.get("confirmed"), bool
                    ):
                        raise ValueError(f"review for {region_id} requires text and confirmed")
                return decoded_result
    except (OSError, json.JSONDecodeError):
        pass
    result: dict[str, dict[str, Any]] = {}
    current: str | None = None
    in_regions = False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read review file: {path}") from error
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and stripped == "regions:":
            in_regions = True
            continue
        if not in_regions:
            raise ValueError("review file must contain a regions mapping")
        if indent == 2 and stripped.endswith(":"):
            current = stripped[:-1].strip()
            result[current] = {}
            continue
        if indent >= 4 and current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key not in {"text", "confirmed"}:
                raise ValueError(f"unsupported review field: {key}")
            result[current][key.strip()] = _parse_scalar(value)
            continue
        raise ValueError(f"malformed review line: {line}")
    if not in_regions:
        raise ValueError("review file must contain a regions mapping")
    for region_id, review in result.items():
        if not isinstance(review.get("text"), str) or not isinstance(review.get("confirmed"), bool):
            raise ValueError(f"review for {region_id} requires text and confirmed")
    return result


def normalize_reading(text: str) -> str:
    """Normalize presentation only; lexical spelling and word choices remain intact."""
    value = re.sub(r"\s+", " ", text.strip())
    value = re.sub(r"\s+[.,]$", "", value)
    if value.startswith("- "):
        value = "- " + value[2:].strip()
    return value


def _successful(group: Any) -> list[dict[str, Any]]:
    if not isinstance(group, dict) or not isinstance(group.get("runs"), list):
        return []
    return [
        item
        for item in group["runs"]
        if isinstance(item, dict)
        and item.get("status") == "ok"
        and isinstance(item.get("text"), str)
    ]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", normalize_reading(text).lower()))


def _candidate(region: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    qwen = _successful(region.get("qwen"))
    trocr = _successful(region.get("trocr"))
    observations = {
        "qwen": [item["text"] for item in qwen],
        "trocr": [item["text"] for item in trocr],
    }
    normalized = {
        name: [normalize_reading(text) for text in texts] for name, texts in observations.items()
    }
    if review is not None:
        text = str(review["text"])
        return {
            "candidate": text,
            "basis": ["human_review"],
            "human_confirmed": review["confirmed"],
            "unresolved": [],
        }

    q_values = list(dict.fromkeys(normalized["qwen"]))
    t_values = list(dict.fromkeys(normalized["trocr"]))
    exact = [value for value in q_values if value in t_values]
    if exact:
        value = exact[0]
        return {
            "candidate": value,
            "basis": ["cross_model_agreement", "stable_recognizer"],
            "human_confirmed": False,
            "unresolved": [],
        }

    # A stable reading can help when the other recognizer shares most of its words,
    # but a stable isolated error is not promoted to a transcription.
    stable_q = len(q_values) == 1 and q_values
    stable_t = len(t_values) == 1 and t_values
    if stable_q and stable_t and _tokens(stable_q[0]) and _tokens(stable_t[0]):
        overlap = len(_tokens(stable_q[0]) & _tokens(stable_t[0])) / max(
            len(_tokens(stable_q[0]) | _tokens(stable_t[0])), 1
        )
        if overlap >= 0.6:
            return {
                "candidate": stable_q[0],
                "basis": ["cross_model_partial_agreement", "stable_recognizer"],
                "human_confirmed": False,
                "unresolved": ["lexical differences remain"],
            }

    if len(q_values) == 1 and not t_values:
        return {
            "candidate": q_values[0],
            "basis": ["qwen_stable"],
            "human_confirmed": False,
            "unresolved": ["TrOCR has no successful observation"],
        }
    if len(t_values) == 1 and not q_values:
        return {
            "candidate": t_values[0],
            "basis": ["trocr_stable"],
            "human_confirmed": False,
            "unresolved": ["Qwen has no successful observation"],
        }
    unresolved = [
        "recognizers disagree" if q_values and t_values else "no successful recognition evidence"
    ]
    return {
        "candidate": "[unclear]",
        "basis": [],
        "human_confirmed": False,
        "unresolved": unresolved,
    }


def _canonical_observation(
    input_path: Path, source: str | None
) -> tuple[dict[str, Any] | None, list[str]]:
    if not source:
        return None, []
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (input_path.parent.parent.parent / source_path).resolve()
    transform = source_path.parent / f"{source_path.stem}.sibyl" / "transform.json"
    if not transform.is_file():
        return None, []
    try:
        value = json.loads(transform.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, []
    refs: list[str] = []
    for region in value.get("regions", []) if isinstance(value, dict) else []:
        if isinstance(region, dict) and region.get("kind") == "figure":
            crop = (
                region.get("source", {}).get("crop")
                if isinstance(region.get("source"), dict)
                else None
            )
            if crop:
                refs.append(f"![Figure {len(refs) + 1}](assets/{Path(crop).name})")
    return {
        "path": str(transform),
        "page_text": value.get("page_text", value.get("interpretation", {}).get("text", [])),
    }, refs


def run_convergence(
    input_path: Path,
    *,
    review_path: Path | None = None,
    markdown_path: Path = DEFAULT_MARKDOWN,
    json_path: Path = DEFAULT_JSON,
) -> dict[str, Any]:
    artifact = _load_json(input_path)
    review = _load_review(review_path)
    regions = artifact["regions"]
    if any(
        not isinstance(region, dict) or not isinstance(region.get("region_id"), str)
        for region in regions
    ):
        raise ValueError("input artifact contains a region without region_id")
    ordered = sorted(
        regions,
        key=lambda region: (
            region.get("crop", {}).get("source_bbox", {}).get("top", 0),
            region["region_id"],
        ),
    )
    converged: list[dict[str, Any]] = []
    for region in ordered:
        region_id = region["region_id"]
        result = _candidate(region, review.get(region_id))
        converged.append(
            {
                "region_id": region_id,
                "source_crop": {
                    key: region.get("crop", {}).get(key)
                    for key in ("path", "sha256", "source_bbox")
                },
                "observations": {
                    name: [
                        {
                            "run": item.get("run"),
                            "status": item.get("status"),
                            "text": item.get("text"),
                        }
                        for item in _successful(region.get(name))
                    ]
                    for name in ("qwen", "trocr")
                },
                "normalized": {
                    name: [
                        normalize_reading(item.get("text", ""))
                        for item in _successful(region.get(name))
                    ]
                    for name in ("qwen", "trocr")
                },
                **result,
            }
        )
    canonical, figures = _canonical_observation(input_path, artifact.get("source"))
    text_blocks = [region["candidate"] for region in converged]
    markdown = "\n\n".join(text_blocks + figures) + "\n"
    output = {
        "experiment": "convergence",
        "input_artifact": str(input_path),
        "review_input": str(review_path) if review_path else None,
        "source": artifact.get("source"),
        "canonical_page_observation": canonical,
        "regions": converged,
        "figures": figures,
        "output": {"markdown": str(markdown_path), "json": str(json_path)},
    }
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def format_convergence_result(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {result['experiment']}",
            f"input: {result['input_artifact']}",
            f"markdown: {result['output']['markdown']}",
            f"json: {result['output']['json']}",
            f"regions: {len(result['regions'])}",
        ]
    )

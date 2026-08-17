"""Deterministic synthesis of preserved recognition evidence (experimental)."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+|[^\w\s]", normalize_reading(text), flags=re.UNICODE)


def _token_similarity(first: str, second: str) -> float:
    if first.lower() == second.lower():
        return 1.0
    if not (first.isalnum() and second.isalnum()) or first[0].lower() != second[0].lower():
        return 0.0
    return SequenceMatcher(None, first.lower(), second.lower(), autojunk=False).ratio()


def _align(template: list[str], reading: list[str]) -> dict[int, list[str]]:
    aligned: dict[int, list[str]] = {}
    matcher = SequenceMatcher(None, template, reading, autojunk=False)
    for tag, left, right, other_left, other_right in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(right - left):
                aligned.setdefault(left + offset, []).append(reading[other_left + offset])
        elif tag == "replace":
            width = min(right - left, other_right - other_left)
            for offset in range(width):
                if _token_similarity(template[left + offset], reading[other_left + offset]) >= 0.5:
                    aligned.setdefault(left + offset, []).append(reading[other_left + offset])
    return aligned


def _template(readings: list[list[str]]) -> list[str]:
    if not readings:
        return []
    scored = [
        (
            sum(
                SequenceMatcher(None, candidate, other, autojunk=False).ratio()
                for other in readings
            ),
            -len(candidate),
            candidate,
        )
        for candidate in readings
    ]
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _detokenize(tokens: list[str]) -> str:
    output = ""
    for token in tokens:
        if token == "[unclear]":
            output += (" " if output and not output.endswith(" ") else "") + token
        elif token in {".", ",", ";", ":", "!", "?", "%", "~"}:
            output = output.rstrip() + token
        elif token == "-" and not output:
            output = "- "
        else:
            output += (" " if output and not output.endswith((" ", "- ")) else "") + token
    return output.strip()


def _recognizer_consensus(readings: list[str]) -> dict[str, Any]:
    token_readings = [_tokenize(reading) for reading in readings if reading.strip()]
    if not token_readings:
        return {"text": "", "tokens": [], "stable_tokens": [], "variants": [], "stable": False}
    template = _template(token_readings)
    aligned = [_align(template, reading) for reading in token_readings]
    threshold = max(1, (len(token_readings) + 1) // 2)
    tokens: list[str] = []
    stable: list[str] = []
    variants: list[dict[str, Any]] = []
    for position, original in enumerate(template):
        observed = [token for item in aligned if position in item for token in item[position]]
        choices = [item for item in observed if _token_similarity(original, item) >= 0.5]
        if len(choices) < threshold:
            tokens.append("[unclear]")
            variants.append({"token": original, "observed": choices, "support": len(choices)})
            continue
        representative = max(
            dict.fromkeys(choices),
            key=lambda item: (
                sum(_token_similarity(item, other) for other in choices),
                choices.count(item),
                -len(item),
            ),
        )
        tokens.append(representative)
        stable.append(representative)
        if len(set(choices)) > 1:
            variants.append({"token": representative, "observed": choices, "support": len(choices)})
    return {
        "text": _detokenize(tokens),
        "tokens": tokens,
        "stable_tokens": stable,
        "variants": variants,
        "stable": len(stable) == len(template) and bool(template),
    }


def _cross_model_matches(first: list[str], second: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for left_index, left in enumerate(first):
        for right_index, right in enumerate(second):
            similarity = _token_similarity(left, right)
            if similarity >= 0.5:
                matches.append(
                    {
                        "qwen_token": left,
                        "trocr_token": right,
                        "similarity": round(similarity, 3),
                        "qwen_index": left_index,
                        "trocr_index": right_index,
                    }
                )
    return matches


def _common_phrases(first: list[str], second: list[str]) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    start: int | None = None
    for index, (left, right) in enumerate(zip(first, second, strict=False)):
        compatible = _token_similarity(left, right) >= 0.5
        if compatible and start is None:
            start = index
        if (not compatible or index == min(len(first), len(second)) - 1) and start is not None:
            end = index if compatible else index - 1
            if end - start + 1 >= 2:
                phrases.append(
                    {
                        "qwen": _detokenize(first[start : end + 1]),
                        "trocr": _detokenize(second[start : end + 1]),
                        "token_count": end - start + 1,
                    }
                )
            start = None
    return phrases


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
    qwen_consensus = _recognizer_consensus(normalized["qwen"])
    trocr_consensus = _recognizer_consensus(normalized["trocr"])
    matches = _cross_model_matches(qwen_consensus["tokens"], trocr_consensus["tokens"])
    common_phrases = _common_phrases(qwen_consensus["tokens"], trocr_consensus["tokens"])
    evidence = {
        "qwen_stability": qwen_consensus,
        "trocr_stability": trocr_consensus,
        "cross_model_overlap": matches,
        "common_phrases": common_phrases,
        "lexical_support": qwen_consensus["stable_tokens"] + trocr_consensus["stable_tokens"],
        "unresolved_tokens": [token for token in qwen_consensus["tokens"] if token == "[unclear]"],
    }
    if review is not None:
        return {
            "candidate": str(review["text"]),
            "basis": ["human_review"],
            "human_confirmed": review["confirmed"],
            "unresolved": [],
            "evidence": evidence,
        }

    q_tokens = list(qwen_consensus["tokens"])
    t_tokens = list(trocr_consensus["tokens"])
    compatible_run = 0
    for q_token, t_token in zip(q_tokens, t_tokens, strict=False):
        if _token_similarity(q_token, t_token) >= 0.5:
            compatible_run += 1
        else:
            break
    if trocr_consensus["stable"] and compatible_run >= 3 and compatible_run >= len(t_tokens) * 0.75:
        return {
            "candidate": trocr_consensus["text"],
            "basis": ["trocr_stability", "cross_model_overlap", "lexical_support"],
            "human_confirmed": False,
            "unresolved": [],
            "evidence": evidence,
        }
    for match in matches:
        index = match["qwen_index"]
        other = t_tokens[match["trocr_index"]]
        if (
            match["similarity"] >= 0.7
            and other != q_tokens[index]
            and any(
                other == variant
                for item in qwen_consensus["variants"]
                for variant in item["observed"]
            )
        ):
            q_tokens[index] = other
    candidate = _detokenize(q_tokens)
    if (
        q_tokens
        and any(token != "[unclear]" for token in q_tokens)
        and (matches or not normalized["trocr"])
    ):
        basis = ["qwen_stability" if qwen_consensus["stable"] else "qwen_phrase_support"]
        if matches:
            basis.append("cross_model_overlap")
        if qwen_consensus["variants"]:
            basis.append("lexical_support")
        return {
            "candidate": candidate,
            "basis": basis,
            "human_confirmed": False,
            "unresolved": evidence["unresolved_tokens"],
            "evidence": evidence,
        }
    if not normalized["qwen"] and trocr_consensus["stable"]:
        return {
            "candidate": trocr_consensus["text"],
            "basis": ["trocr_stability"],
            "human_confirmed": False,
            "unresolved": [],
            "evidence": evidence,
        }
    return {
        "candidate": "[unclear]",
        "basis": [],
        "human_confirmed": False,
        "unresolved": [
            "recognizers disagree"
            if q_tokens and trocr_consensus["tokens"]
            else "no successful recognition evidence"
        ],
        "evidence": evidence,
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

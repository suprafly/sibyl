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


def _source_bbox(region: dict[str, Any]) -> dict[str, Any]:
    value = region.get("source_crop", {}).get("source_bbox", {})
    return value if isinstance(value, dict) else {}


def _document_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        regions,
        key=lambda region: (
            _source_bbox(region).get("top", 0),
            _source_bbox(region).get("left", 0),
            region["region_id"],
        ),
    )


def _candidate_tokens(candidate: str) -> list[str]:
    return [token for token in _tokenize(candidate) if token != "[unclear]"]


def _same_line(first: dict[str, Any], second: dict[str, Any]) -> bool:
    left = _source_bbox(first)
    right = _source_bbox(second)
    try:
        first_top, first_bottom = float(left["top"]), float(left["bottom"])
        second_top, second_bottom = float(right["top"]), float(right["bottom"])
        first_height = max(1.0, first_bottom - first_top)
        second_height = max(1.0, second_bottom - second_top)
    except (KeyError, TypeError, ValueError):
        return False
    overlap = min(first_bottom, second_bottom) - max(first_top, second_top)
    return overlap >= 0.5 * min(first_height, second_height)


def _continuity_support(candidate: str, neighbors: list[str]) -> int:
    candidate_tokens = _candidate_tokens(candidate)
    score = 0
    for neighbor in neighbors:
        neighbor_tokens = _candidate_tokens(neighbor)
        score += sum(
            1
            for left, right in zip(candidate_tokens, neighbor_tokens, strict=False)
            if _token_similarity(left, right) >= 0.7 and left.isalnum() and right.isalnum()
        )
    return score


def _page_extensions(candidate: str, page_lines: list[str]) -> list[str]:
    source = _tokenize(candidate)
    if len(source) < 2:
        return []
    extensions: list[str] = []
    for line in page_lines:
        page = _tokenize(line)
        best: tuple[int, int] | None = None
        for source_start in range(len(source) - 1):
            suffix = source[source_start:]
            for page_start in range(len(page) - len(suffix) + 1):
                matched = sum(
                    _token_similarity(left, right) >= 0.7
                    for left, right in zip(suffix, page[page_start:], strict=False)
                )
                if (
                    matched == len(suffix)
                    and page_start + len(suffix) < len(page)
                    and (best is None or len(suffix) > best[0])
                ):
                    best = (len(suffix), page_start + len(suffix))
        if best is not None:
            extensions.append(_detokenize(source + page[best[1] :]))
    return extensions


def _score_document_candidate(
    region: dict[str, Any],
    candidate: str,
    *,
    selected_neighbors: list[str],
    page_lines: list[str],
) -> dict[str, Any]:
    normalized_candidate = normalize_reading(candidate)
    observations = region["observations"]
    qwen = [normalize_reading(item["text"]) for item in observations["qwen"]]
    trocr = [normalize_reading(item["text"]) for item in observations["trocr"]]
    recognition = [
        name
        for name, values in (("qwen", qwen), ("trocr", trocr))
        if normalized_candidate in values
    ]
    cross_model = [
        match
        for match in region.get("evidence", {}).get("cross_model_overlap", [])
        if match.get("qwen_token") in _candidate_tokens(candidate)
        or match.get("trocr_token") in _candidate_tokens(candidate)
    ]
    local_stability = []
    for name in ("qwen_stability", "trocr_stability"):
        stability = region.get("evidence", {}).get(name, {})
        if normalized_candidate == stability.get("text"):
            local_stability.append(name)
    page_support = [line for line in page_lines if normalize_reading(line) == normalized_candidate]
    page_token_support = max(
        (
            sum(
                any(_token_similarity(token, page_token) >= 0.7 for page_token in _tokenize(line))
                for token in _candidate_tokens(candidate)
            )
            for line in page_lines
        ),
        default=0,
    )
    continuity = _continuity_support(candidate, selected_neighbors)
    observed_tokens = {
        token.lower()
        for value in qwen + trocr
        for token in _candidate_tokens(value)
        if token.isalnum()
    }
    page_tokens = {
        token.lower()
        for value in page_lines
        for token in _candidate_tokens(value)
        if token.isalnum()
    }
    unsupported = [
        token
        for token in _candidate_tokens(candidate)
        if token.isalnum() and token.lower() not in observed_tokens | page_tokens
    ]
    score = (
        len(recognition)
        + len(cross_model)
        + len(local_stability)
        + 3 * continuity
        + 4 * len(page_support)
        + 3 * page_token_support
        - 3 * len(unsupported)
    )
    return {
        "score": score,
        "basis": {
            "recognition_support": recognition,
            "cross_model_support": cross_model,
            "local_stability": local_stability,
            "lexical_continuity": continuity,
            "spatial_continuity": [],
            "page_level_support": page_support,
            "page_level_token_support": page_token_support,
            "conflicts": unsupported,
        },
    }


def _document_convergence(
    regions: list[dict[str, Any]], page_lines: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = _document_order(regions)
    selected: list[str] = []
    decisions: list[dict[str, Any]] = []
    for region in ordered:
        alternatives = list(dict.fromkeys([region["candidate"], *region.get("alternatives", [])]))
        regional_first_tokens = [
            tokens[0]
            for candidate in alternatives
            if (tokens := _candidate_tokens(candidate)) and tokens[0].isalnum()
        ]
        regional_tokens = {
            token.lower()
            for candidate in alternatives
            for token in _candidate_tokens(candidate)
            if token.isalnum()
        }
        max_regional_length = max(
            (len(_candidate_tokens(candidate)) for candidate in alternatives), default=0
        )
        for line in page_lines:
            page_tokens = [token for token in _candidate_tokens(line) if token.isalnum()]
            overlap = sum(
                1
                for token in page_tokens
                if token.lower() in regional_tokens
                or any(_token_similarity(token, other) >= 0.7 for other in regional_tokens)
            )
            comparable_length = min(len(page_tokens), max_regional_length)
            first_token_matches = bool(page_tokens) and any(
                _token_similarity(page_tokens[0], token) >= 0.7 for token in regional_first_tokens
            )
            if overlap >= 3 and overlap * 10 >= 6 * comparable_length and first_token_matches:
                alternatives.append(line)
        alternatives.extend(
            extension
            for candidate in alternatives
            for extension in _page_extensions(candidate, page_lines)
        )
        alternatives = list(dict.fromkeys(alternatives))
        alternatives = [candidate for candidate in alternatives if isinstance(candidate, str)]
        neighboring = selected[-2:]
        scores = [
            {
                "candidate": candidate,
                **_score_document_candidate(
                    region,
                    candidate,
                    selected_neighbors=neighboring,
                    page_lines=page_lines,
                ),
            }
            for candidate in alternatives
        ]
        human_confirmed = bool(region.get("human_confirmed"))
        if human_confirmed:
            chosen = region["candidate"]
            chosen_score = next(item for item in scores if item["candidate"] == chosen)
        else:
            viable = [
                item
                for item in scores
                if item["candidate"] != "[unclear]"
                and not item["basis"]["conflicts"]
                and (
                    item["candidate"] == region["candidate"]
                    or item["basis"]["cross_model_support"]
                    or item["basis"]["page_level_support"]
                    or item["basis"]["page_level_token_support"] >= 2
                    or item["basis"]["lexical_continuity"]
                )
            ]
            chosen_score = (
                max(
                    viable, key=lambda item: (item["score"], -alternatives.index(item["candidate"]))
                )
                if viable
                else next(item for item in scores if item["candidate"] == "[unclear]")
            )
            chosen = chosen_score["candidate"]
        selected.append(chosen)
        decisions.append(
            {
                "region_id": region["region_id"],
                "regional_candidate": region["candidate"],
                "alternatives": alternatives,
                "selected": chosen,
                "score": chosen_score["score"],
                "basis": chosen_score["basis"],
                "human_confirmed": human_confirmed,
                "source_crop": region["source_crop"],
            }
        )
    blocks: list[str] = []
    for index, decision in enumerate(decisions):
        if index and _same_line(ordered[index - 1], ordered[index]):
            blocks[-1] = f"{blocks[-1]} {decision['selected']}".strip()
        else:
            blocks.append(decision["selected"])
    return decisions, blocks


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
            region.get("crop", {}).get("source_bbox", {}).get("left", 0),
            region["region_id"],
        ),
    )
    converged: list[dict[str, Any]] = []
    for region in ordered:
        region_id = region["region_id"]
        result = _candidate(region, review.get(region_id))
        result["alternatives"] = list(
            dict.fromkeys(
                [
                    normalize_reading(item.get("text", ""))
                    for name in ("qwen", "trocr")
                    for item in _successful(region.get(name))
                ]
                + [result["candidate"]]
            )
        )
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
    page_lines = []
    if isinstance(canonical, dict) and isinstance(canonical.get("page_text"), list):
        page_lines = [line for line in canonical["page_text"] if isinstance(line, str)]
    document_regions, text_blocks = _document_convergence(converged, page_lines)
    markdown = "\n\n".join(text_blocks + figures) + "\n"
    output = {
        "experiment": "convergence",
        "input_artifact": str(input_path),
        "review_input": str(review_path) if review_path else None,
        "source": artifact.get("source"),
        "canonical_page_observation": canonical,
        "regions": converged,
        "document_candidate": {
            "regions": [item["selected"] for item in document_regions],
            "figures": figures,
        },
        "document_convergence": {"regions": document_regions, "blocks": text_blocks},
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

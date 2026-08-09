from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "maxim_9b_answer_contract_repair_v1_1"
MODEL = "Qwen/Qwen3.5-9B"
ROWS = 274
PROFILE_SCHEMA = "maxim-9b-answer-contract-repair-profile-v1.1"
RULE_FREEZE_SCHEMA = "maxim-9b-answer-contract-repair-rule-freeze-v1.1"
DECISION_SCHEMA = "maxim-9b-answer-contract-repair-decision-v1.1"
MANIFEST_SCHEMA = "maxim-9b-answer-contract-repair-candidate-manifest-v1.1"
OUTPUT_FREEZE_SCHEMA = "maxim-9b-answer-contract-repair-output-freeze-v1.1"
ROW_PROVENANCE_SCHEMA = "maxim-9b-answer-contract-repair-row-provenance-v1.1"

MAX_RAW_RESPONSE_BYTES = 262_144
MAX_ANSWER_CHARS = 256
MAX_EXPLORATORY_VALUE_CHARS = 64
MAX_OUTER_MEMBERS = 64
ARMS = ("strict", "exploratory_explicit_key_scalar")
EXPLORATORY_KEYS = ("final_answer", "answer", "choice", "result")
BASE_VARIANTS = ("on_final_9b_source", "on_v1_2_primary_240")
BASE_INPUT_KEYS = {
    "on_final_9b_source": "base_solver",
    "on_v1_2_primary_240": "v1_2_primary_240_solver",
}

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = EXPERIMENT_ROOT / "profile.json"
RULE_FREEZE_PATH = EXPERIMENT_ROOT / "DEVELOPMENT_RULE_FREEZE.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "candidate_output"
SUPERSEDED_RECORD: dict[str, str] = {
    "path": "experiments/maxim_9b_answer_contract_repair_v1/SUPERSEDED.json",
    "sha256": "ef597a82f61138d586ec352e063e7ea1a73c2c4b7d4c4278752af70d774aa197",
}

INPUTS: dict[str, dict[str, str]] = {
    "base_solver": {
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/fill_composed/solver.jsonl",
        "sha256": "9d26067064ee07fe480391759782c86d66adbb76dbc0da0d86ccc1b3f035211e",
    },
    "v1_2_primary_240_solver": {
        "path": "experiments/maxim_9b_baseline_selector_v1/compositor_output_v1_2/primary_solver.jsonl",
        "sha256": "09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef",
    },
    "benchmark_order": {
        "path": "experiments/maxim_9b_baseline_selector_v1/input/frozen/benchmark_order_v1_1.json",
        "sha256": "7140c7c01b48053f6a15a3b0113f68cad37bbb887744828b570a6eaa0447d62b",
    },
    "route_map": {
        "path": "experiments/maxim_9b_baseline_selector_v1/input/frozen/evaluator_route_map_v1_1.json",
        "sha256": "f89ef00f95b9d83610b66948fcb11667dc927f2452b000ef62e031a1a0de26f6",
    },
    "source_union_membership": {
        "path": "experiments/maxim_9b_baseline_selector_v1/input/frozen/source_union_membership_v1_1.json",
        "sha256": "93a1018a63e2b9dfeef841541df3b566d6bd6275471accb9f167c7c60c44416a",
    },
}

FORBIDDEN_KEYS = frozenset(
    {
        "accuracy",
        "correct",
        "correctness",
        "gold",
        "gold_answer",
        "is_correct",
        "judge",
        "judge_verdict",
        "outcome",
        "outcomes",
        "reference_answer",
        "reference_solution",
        "reward",
        "score",
        "scores",
        "strict_correct",
        "verdict",
    }
)
FORBIDDEN_BYTES = (
    b"qwen/qwen3.5-27b",
    b"qwen3.5-27b",
    b'"inherited_27b_outputs":true',
)

EXPECTED_PROFILE: dict[str, Any] = {
    "schema_version": PROFILE_SCHEMA,
    "experiment_id": EXPERIMENT_ID,
    "status": "post_score_motivated_development_rule_not_blind_not_preregistered",
    "chronology": {
        "historical_residual_outcomes_known_before_design": True,
        "post_score_motivated": True,
        "blind_claim": False,
        "preregistered_claim": False,
        "rules_must_be_frozen_before_candidate_materialization": True,
        "candidate_must_remain_unscored_and_unevaluated": True,
    },
    "model_closure": [MODEL],
    "runtime_outcome_access": False,
    "inputs": INPUTS,
    "scope": {
        "preserve_source_union_rows_as_exact_base_line_bytes": True,
        "preserve_image_judge_rows_as_exact_base_line_bytes": True,
        "eligible_route": "deterministic",
        "eligible_membership": "outside_pinned_source_union",
        "eligible_answer_state": "top_level_final_answer_absent_or_strictly_invalid",
        "task_id_specific_rules": False,
        "base_variants": list(BASE_VARIANTS),
    },
    "strict_answer_contract": {
        "accepted_scalar_types": ["string", "integer", "finite_number"],
        "normalize": "unicode_preserving_outer_whitespace_strip_only",
        "max_characters": MAX_ANSWER_CHARS,
        "reject_empty": True,
        "reject_control_characters": True,
        "reject_json_container_or_quoted_json_prefix": True,
        "require_balanced_round_square_curly_delimiters": True,
    },
    "arms": {
        "strict": {
            "source_field": "raw_response",
            "parser": "context_aware_exact_outer_response_object_members_only",
            "candidate_key": "final_answer",
            "max_raw_response_bytes": MAX_RAW_RESPONSE_BYTES,
            "max_outer_members": MAX_OUTER_MEMBERS,
            "accepted_outer_parse_modes": [
                "exact_json_object_preserving_duplicate_keys"
            ],
            "exactly_one_outer_final_answer_key_required": True,
            "json_string_values_are_atomic": True,
            "keys_inside_reasoning_or_any_other_value_are_forbidden": True,
            "unique_distinct_strict_candidate_required": True,
            "zero_multiple_invalid_or_parse_failure": "preserve_base_exact_bytes",
            "reasoning_text_regex_or_numeric_salvage": False,
        },
        "exploratory_explicit_key_scalar": {
            "label": "explicitly_post_score_motivated_exploratory_not_blind_not_preregistered",
            "source_field": "raw_response",
            "parser": "context_aware_outer_response_object_members_only",
            "candidate_keys": list(EXPLORATORY_KEYS),
            "exactly_one_key_occurrence_required": True,
            "max_raw_response_bytes": MAX_RAW_RESPONSE_BYTES,
            "max_value_characters_before_and_after_normalization": MAX_EXPLORATORY_VALUE_CHARS,
            "max_outer_members": MAX_OUTER_MEMBERS,
            "accepted_outer_parse_modes": [
                "exact_json_object_preserving_duplicate_keys",
                "missing_final_outer_curly_only",
            ],
            "json_string_values_are_atomic": True,
            "keys_inside_reasoning_or_any_other_value_are_forbidden": True,
            "global_unescape_or_key_search_over_value_text": False,
            "normalization": "strip_only_unmatched_leading_closing_or_trailing_opening_square_curly_braces_from_the_single_top_level_scalar_value",
            "unique_distinct_strict_candidate_required": True,
            "zero_multiple_ambiguous_or_parse_failure": "preserve_base_exact_bytes",
            "free_reasoning_or_unkeyed_number_mining": False,
        },
    },
    "mutation": {
        "copy_full_base_row": True,
        "replace_only_top_level_final_answer": True,
        "retain_raw_response": True,
        "add_explicit_generation_provenance": True,
        "no_fallback_repair": True,
        "fail_closed": True,
    },
    "supersedes": {
        "experiment_id": "maxim_9b_answer_contract_repair_v1",
        "record": SUPERSEDED_RECORD,
        "reason": "independent_audit_rejected_global_unescape_then_regex_as_not_value_context_safe",
    },
}


class RepairError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RepairError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RepairError(f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairError(f"{label} invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"{label} must be a JSON object")
    return value


def _read_jsonl_raw(path: Path, label: str) -> tuple[list[dict[str, Any]], list[bytes]]:
    try:
        lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
    except OSError as exc:
        raise RepairError(f"cannot read {label}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines):
        if not raw.endswith(b"\n"):
            raise RepairError(f"{label}:{index}: missing terminating newline")
        try:
            row = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RepairError(f"{label}:{index}: invalid JSON row: {exc}") from exc
        if not isinstance(row, dict):
            raise RepairError(f"{label}:{index}: row must be object")
        rows.append(row)
    return rows, lines


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.write_bytes(canonical_json(value))
    return sha256_file(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json(row))
    return sha256_file(path)


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk(item)


def _assert_no_outcome_or_mixed_model(value: Any, label: str) -> None:
    for key, item in _walk(value):
        lowered = key.casefold() if key is not None else ""
        if lowered in FORBIDDEN_KEYS:
            raise RepairError(f"{label}: forbidden field {key}")
        if lowered == "gold_access" and item is not False:
            raise RepairError(f"{label}: non-false gold_access")
        if lowered == "model" and item not in (None, MODEL):
            raise RepairError(f"{label}: model closure violation {item!r}")
        if lowered == "inherited_27b_outputs" and item is not False:
            raise RepairError(f"{label}: inherited 27B marker")


def _assert_no_27b_bytes(path: Path, label: str) -> None:
    lowered = path.read_bytes().lower().replace(b" ", b"")
    for marker in FORBIDDEN_BYTES:
        if marker in lowered:
            raise RepairError(f"{label}: forbidden 27B marker {marker!r}")


def strict_answer(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text or len(text) > MAX_ANSWER_CHARS:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    if text[0] in '{["':
        return None
    opening = {"(", "[", "{"}
    closing = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in text:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return None
    if stack:
        return None
    return text


_FENCE_RE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z", re.I)


def recover_unique_answer(raw_response: Any) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        return {
            "status": "missing_raw_response",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": None,
        }
    if len(raw_response.encode("utf-8")) > MAX_RAW_RESPONSE_BYTES:
        return {
            "status": "bounds_exceeded",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": None,
        }
    parsed = _outer_object_members(raw_response)
    parse_mode = parsed["status"]
    if parse_mode != "exact_json_object_preserving_duplicate_keys":
        return {
            "status": parse_mode,
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": parse_mode,
        }
    matches = [value for key, value in parsed["members"] if key == "final_answer"]
    if not matches:
        return {
            "status": "no_explicit_outer_final_answer",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": parse_mode,
        }
    if len(matches) != 1:
        return {
            "status": "ambiguous_multiple_outer_final_answer_keys",
            "candidates": [],
            "key_occurrences": len(matches),
            "outer_parse_mode": parse_mode,
        }
    candidate = strict_answer(matches[0])
    if candidate is None:
        return {
            "status": "outer_final_answer_not_strict",
            "candidates": [],
            "key_occurrences": 1,
            "outer_parse_mode": parse_mode,
        }
    return {
        "status": "unique_strict_candidate",
        "candidates": [candidate],
        "key_occurrences": 1,
        "outer_parse_mode": parse_mode,
    }


class _ObjectPairs(list[tuple[str, Any]]):
    """Distinct type used to preserve duplicate object keys during JSON decode."""


def _json_body(text: str) -> str | None:
    stripped = text.strip()
    fence = _FENCE_RE.fullmatch(stripped)
    if fence:
        stripped = fence.group("body").strip()
    return stripped or None


def _outer_object_members(raw_response: str) -> dict[str, Any]:
    """Parse only actual outer members; JSON string values always stay atomic."""
    body = _json_body(raw_response)
    if body is None or not body.startswith("{"):
        return {"status": "outer_object_parse_failure", "members": []}
    decoder = json.JSONDecoder(
        object_pairs_hook=_ObjectPairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"nonfinite constant {token}")
        ),
    )
    try:
        exact = decoder.decode(body)
    except (json.JSONDecodeError, ValueError, TypeError):
        exact = None
    if isinstance(exact, _ObjectPairs):
        if len(exact) > MAX_OUTER_MEMBERS:
            return {"status": "bounds_exceeded", "members": []}
        return {
            "status": "exact_json_object_preserving_duplicate_keys",
            "members": list(exact),
        }
    if exact is not None:
        return {"status": "outer_response_is_not_object", "members": []}

    # The sole malformed-outer repair is a missing final `}`. Every key and value
    # still has to be a complete JSON token. In particular, a reasoning string is
    # consumed atomically and its contents are never rescanned for key-like text.
    position = 1
    members: list[tuple[str, Any]] = []
    length = len(body)
    while True:
        while position < length and body[position] in " \t\r\n":
            position += 1
        if position >= length:
            return {
                "status": "missing_final_outer_curly_only",
                "members": members,
            }
        if body[position] == "}":
            return {"status": "outer_object_parse_failure", "members": []}
        try:
            key, consumed = json.JSONDecoder().raw_decode(body[position:])
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"status": "outer_object_parse_failure", "members": []}
        if not isinstance(key, str):
            return {"status": "outer_object_parse_failure", "members": []}
        position += consumed
        while position < length and body[position] in " \t\r\n":
            position += 1
        if position >= length or body[position] != ":":
            return {"status": "outer_object_parse_failure", "members": []}
        position += 1
        while position < length and body[position] in " \t\r\n":
            position += 1
        if position >= length:
            return {"status": "outer_object_parse_failure", "members": []}
        try:
            value, consumed = decoder.raw_decode(body[position:])
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"status": "outer_object_parse_failure", "members": []}
        position += consumed
        members.append((key, value))
        if len(members) > MAX_OUTER_MEMBERS:
            return {"status": "bounds_exceeded", "members": []}
        while position < length and body[position] in " \t\r\n":
            position += 1
        if position >= length:
            return {
                "status": "missing_final_outer_curly_only",
                "members": members,
            }
        if body[position] != ",":
            return {"status": "outer_object_parse_failure", "members": []}
        position += 1
        lookahead = position
        while lookahead < length and body[lookahead] in " \t\r\n":
            lookahead += 1
        if lookahead >= length:
            return {"status": "outer_object_parse_failure", "members": []}


def _normalize_exploratory_scalar(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if not text or len(text) > MAX_EXPLORATORY_VALUE_CHARS:
            return None
        # Only discard structurally impossible edge debris. Matched wrappers and
        # interior braces are deliberately left untouched and fail strict_answer.
        text = re.sub(r"\A(?:[}\]]\s*)+", "", text)
        text = re.sub(r"(?:\s*[{\[])\Z", "", text).strip()
        if not text or len(text) > MAX_EXPLORATORY_VALUE_CHARS:
            return None
        return strict_answer(text)
    candidate = strict_answer(value)
    if candidate is None or len(candidate) > MAX_EXPLORATORY_VALUE_CHARS:
        return None
    return candidate


def recover_exploratory_answer(raw_response: Any) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        return {
            "status": "missing_raw_response",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": None,
        }
    if len(raw_response.encode("utf-8")) > MAX_RAW_RESPONSE_BYTES:
        return {
            "status": "bounds_exceeded",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": None,
        }
    parsed = _outer_object_members(raw_response)
    parse_mode = parsed["status"]
    if parse_mode not in {
        "exact_json_object_preserving_duplicate_keys",
        "missing_final_outer_curly_only",
    }:
        return {
            "status": parse_mode,
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": parse_mode,
        }
    matches = [(key, value) for key, value in parsed["members"] if key in EXPLORATORY_KEYS]
    if not matches:
        return {
            "status": "no_explicit_outer_key",
            "candidates": [],
            "key_occurrences": 0,
            "outer_parse_mode": parse_mode,
        }
    if len(matches) != 1:
        return {
            "status": "ambiguous_multiple_explicit_keys",
            "candidates": [],
            "key_occurrences": len(matches),
            "outer_parse_mode": parse_mode,
        }
    key, value = matches[0]
    candidate = _normalize_exploratory_scalar(value)
    if candidate is None:
        return {
            "status": "explicit_key_value_not_strict_or_out_of_bounds",
            "candidates": [],
            "key_occurrences": 1,
            "outer_parse_mode": parse_mode,
        }
    return {
        "status": "unique_explicit_key_scalar_candidate",
        "candidates": [candidate],
        "key_occurrences": 1,
        "matched_key": key,
        "outer_parse_mode": parse_mode,
    }


def decide_row(
    row: Mapping[str, Any], *, evaluation_route: str, protected_by_source_union: bool,
    arm: str = "strict",
) -> dict[str, Any]:
    if arm not in ARMS:
        raise RepairError(f"unknown repair arm {arm!r}")
    if protected_by_source_union:
        return {
            "action": "preserve_base_exact_bytes",
            "reason": "protected_by_source_union",
            "top_level_answer_valid": strict_answer(row.get("final_answer")) is not None,
            "parser_status": "not_run_protected",
            "candidates": [],
        }
    if evaluation_route == "image_judge":
        return {
            "action": "preserve_base_exact_bytes",
            "reason": "image_judge_route",
            "top_level_answer_valid": strict_answer(row.get("final_answer")) is not None,
            "parser_status": "not_run_image",
            "candidates": [],
        }
    if evaluation_route != "deterministic":
        raise RepairError(f"unknown evaluation route {evaluation_route!r}")
    if strict_answer(row.get("final_answer")) is not None:
        return {
            "action": "preserve_base_exact_bytes",
            "reason": "top_level_answer_valid",
            "top_level_answer_valid": True,
            "parser_status": "not_run_valid",
            "candidates": [],
        }
    if arm == "strict":
        recovered = recover_unique_answer(row.get("raw_response"))
        unique_status = "unique_strict_candidate"
        reason = "unique_strict_outer_final_answer"
    else:
        recovered = recover_exploratory_answer(row.get("raw_response"))
        unique_status = "unique_explicit_key_scalar_candidate"
        reason = "unique_explicit_key_scalar_candidate_exploratory"
    if recovered["status"] == unique_status:
        return {
            "action": "repair_from_raw_response",
            "reason": reason,
            "top_level_answer_valid": False,
            "parser_status": recovered["status"],
            "candidates": recovered["candidates"],
        }
    return {
        "action": "preserve_base_exact_bytes",
        "reason": recovered["status"],
        "top_level_answer_valid": False,
        "parser_status": recovered["status"],
        "candidates": recovered["candidates"],
    }


def _pinned_path(descriptor: Mapping[str, str], label: str) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RepairError(f"{label}: unsafe path")
    path = REPO_ROOT / relative
    if sha256_file(path) != descriptor["sha256"]:
        raise RepairError(f"{label}: SHA-256 mismatch")
    return path


def _verify_superseded_record() -> dict[str, Any]:
    record_path = _pinned_path(SUPERSEDED_RECORD, "superseded predecessor record")
    record = _read_json(record_path, "superseded predecessor record")
    if record.get("status") != (
        "superseded_before_evaluation_due_to_independent_parser_contract_audit_failure"
    ):
        raise RepairError("predecessor superseded status mismatch")
    if record.get("evaluated") is not False or record.get("scored") is not False:
        raise RepairError("superseded predecessor was evaluated or scored")
    preserved = record.get("preserved_artifacts")
    if not isinstance(preserved, dict) or len(preserved) != 5:
        raise RepairError("superseded predecessor artifact closure mismatch")
    for label, descriptor in preserved.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise RepairError(f"superseded predecessor descriptor mismatch: {label}")
        relative = Path(descriptor["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RepairError(f"superseded predecessor unsafe path: {label}")
        artifact = record_path.parent / relative
        if sha256_file(artifact) != descriptor["sha256"]:
            raise RepairError(f"superseded predecessor SHA mismatch: {label}")
    return record


def _load_inputs(base_variant: str = "on_final_9b_source") -> dict[str, Any]:
    if base_variant not in BASE_VARIANTS:
        raise RepairError(f"unknown base variant {base_variant!r}")
    paths = {key: _pinned_path(descriptor, key) for key, descriptor in INPUTS.items()}
    base_key = BASE_INPUT_KEYS[base_variant]
    base_path = paths[base_key]
    _assert_no_27b_bytes(base_path, f"{base_variant} base solver")
    base_rows, base_raw = _read_jsonl_raw(base_path, f"{base_variant} base solver")
    order_doc = _read_json(paths["benchmark_order"], "benchmark order")
    route_doc = _read_json(paths["route_map"], "route map")
    membership_doc = _read_json(paths["source_union_membership"], "source union")
    order = order_doc.get("rows")
    route_rows = route_doc.get("rows")
    protected_ids = membership_doc.get("task_ids")
    if not isinstance(order, list) or len(order) != ROWS or len(set(order)) != ROWS:
        raise RepairError("benchmark order is not 274 unique task IDs")
    if not isinstance(route_rows, list) or len(route_rows) != ROWS:
        raise RepairError("route map is not full274")
    if not isinstance(protected_ids, list) or len(protected_ids) != 156 or len(set(protected_ids)) != 156:
        raise RepairError("source union is not 156 unique task IDs")
    routes: list[str] = []
    for index, (task_id, route_row) in enumerate(zip(order, route_rows, strict=True)):
        expected = {
            "row_index": index,
            "task_id": task_id,
            "evaluation_route": route_row.get("evaluation_route"),
        }
        if route_row != expected or route_row.get("evaluation_route") not in {
            "deterministic",
            "image_judge",
        }:
            raise RepairError(f"route map misalignment at row {index}")
        routes.append(route_row["evaluation_route"])
    if Counter(routes) != {"deterministic": 177, "image_judge": 97}:
        raise RepairError("route split changed")
    if len(base_rows) != ROWS or [row.get("task_id") for row in base_rows] != order:
        raise RepairError("base solver order differs from authority")
    for index, row in enumerate(base_rows):
        if row.get("model") != MODEL:
            raise RepairError(f"base row {index}: model is not Qwen3.5-9B")
        generation = row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise RepairError(f"base row {index}: gold_access is not false")
        _assert_no_outcome_or_mixed_model(row, f"base row {index}")
    return {
        "paths": paths,
        "base_variant": base_variant,
        "base_input_key": base_key,
        "base_rows": base_rows,
        "base_raw": base_raw,
        "order": order,
        "routes": routes,
        "protected": frozenset(protected_ids),
    }


def _profile() -> dict[str, Any]:
    _verify_superseded_record()
    profile = _read_json(PROFILE_PATH, "repair profile")
    if profile != EXPECTED_PROFILE:
        raise RepairError("profile differs from code-frozen rules")
    return profile


def _rule_freeze_payload() -> dict[str, Any]:
    return {
        "schema_version": RULE_FREEZE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "development_rules_frozen_before_candidate_build_not_blind_not_preregistered",
        "chronology": {
            "historical_residual_outcomes_known_before_design": True,
            "post_score_motivated": True,
            "blind_claim": False,
            "preregistered_claim": False,
            "frozen_before_candidate_materialization": True,
            "candidate_output_absent_at_freeze": True,
        },
        "runtime_outcome_access": False,
        "artifacts": {
            "profile": {"path": "profile.json", "sha256": sha256_file(PROFILE_PATH)},
            "code": {
                "path": "answer_contract_repair_v1_1.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "tests": {
                "path": "test_answer_contract_repair_v1_1.py",
                "sha256": sha256_file(
                    EXPERIMENT_ROOT / "test_answer_contract_repair_v1_1.py"
                ),
            },
            "readme": {
                "path": "README.md",
                "sha256": sha256_file(EXPERIMENT_ROOT / "README.md"),
            },
            "superseded_predecessor_record": SUPERSEDED_RECORD,
        },
        "inputs": INPUTS,
        "supersedes": EXPECTED_PROFILE["supersedes"],
    }


def write_rule_freeze() -> dict[str, Any]:
    _profile()
    for base_variant in BASE_VARIANTS:
        _load_inputs(base_variant)
    if RULE_FREEZE_PATH.exists():
        raise RepairError("rule freeze already exists; refusing overwrite")
    if DEFAULT_OUTPUT.exists():
        raise RepairError("candidate output exists before rule freeze")
    payload = _rule_freeze_payload()
    temporary = RULE_FREEZE_PATH.with_suffix(".tmp")
    temporary.write_bytes(canonical_json(payload))
    os.replace(temporary, RULE_FREEZE_PATH)
    return verify_rule_freeze()


def verify_rule_freeze() -> dict[str, Any]:
    _profile()
    freeze = _read_json(RULE_FREEZE_PATH, "development rule freeze")
    if freeze != _rule_freeze_payload():
        raise RepairError("development rule freeze payload/hash closure mismatch")
    if freeze["runtime_outcome_access"] is not False:
        raise RepairError("rule freeze runtime outcome access is not false")
    return {
        "status": "development_rule_freeze_verified",
        "path": str(RULE_FREEZE_PATH),
        "sha256": sha256_file(RULE_FREEZE_PATH),
    }


def _plan_candidate(
    rule_freeze_sha256: str, arm: str, base_variant: str = "on_final_9b_source"
) -> dict[str, Any]:
    if arm not in ARMS:
        raise RepairError(f"unknown repair arm {arm!r}")
    bound = _load_inputs(base_variant)
    output_lines: list[bytes] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    invalid_eligible = 0
    for index, task_id in enumerate(bound["order"]):
        base = bound["base_rows"][index]
        base_raw = bound["base_raw"][index]
        route = bound["routes"][index]
        protected = task_id in bound["protected"]
        decision = decide_row(
            base,
            evaluation_route=route,
            protected_by_source_union=protected,
            arm=arm,
        )
        if route == "deterministic" and not protected and not decision["top_level_answer_valid"]:
            invalid_eligible += 1
        if decision["action"] == "repair_from_raw_response":
            candidate = decision["candidates"][0]
            changed = copy.deepcopy(base)
            generation = changed.get("generation")
            if not isinstance(generation, dict):
                raise RepairError(f"row {index}: generation missing")
            changed["final_answer"] = candidate
            generation["answer_contract_repair_v1_1"] = {
                "schema_version": ROW_PROVENANCE_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "arm": arm,
                "base_variant": base_variant,
                "rule_freeze_sha256": rule_freeze_sha256,
                "action": (
                    "replace_invalid_top_level_answer_from_unique_outer_final_answer_candidate"
                    if arm == "strict"
                    else "replace_invalid_top_level_answer_from_unique_explicit_key_scalar_candidate_exploratory"
                ),
                "base_row_sha256": sha256_bytes(base_raw),
                "raw_response_sha256": sha256_bytes(
                    str(base.get("raw_response") or "").encode("utf-8")
                ),
                "selected_answer_sha256": sha256_bytes(candidate.encode("utf-8")),
                "candidate_count": 1,
                "authoritative_route": route,
                "protected_by_source_union": False,
                "runtime_outcome_access": False,
            }
            _assert_no_outcome_or_mixed_model(changed, f"candidate row {index}")
            output_raw = canonical_json(changed)
            counts["repair_from_raw_response"] += 1
        else:
            output_raw = base_raw
            counts[decision["reason"]] += 1
        output_lines.append(output_raw)
        decisions.append(
            {
                "schema_version": DECISION_SCHEMA,
                "arm": arm,
                "base_variant": base_variant,
                "row_index": index,
                "task_id": task_id,
                "authoritative_route": route,
                "protected_by_source_union": protected,
                "top_level_answer_valid": decision["top_level_answer_valid"],
                "parser_status": decision["parser_status"],
                "candidate_count": len(decision["candidates"]),
                "action": decision["action"],
                "reason": decision["reason"],
                "base_row_sha256": sha256_bytes(base_raw),
                "output_row_sha256": sha256_bytes(output_raw),
                "runtime_outcome_access": False,
            }
        )
    return {
        **bound,
        "output_lines": output_lines,
        "decisions": decisions,
        "counts": counts,
        "invalid_eligible": invalid_eligible,
    }


def build_candidate(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists():
        raise RepairError(f"refusing to overwrite candidate output {output_dir}")
    freeze_report = verify_rule_freeze()
    rule_freeze_sha = freeze_report["sha256"]
    plans = {
        base_variant: {
            arm: _plan_candidate(rule_freeze_sha, arm, base_variant)
            for arm in ARMS
        }
        for base_variant in BASE_VARIANTS
    }
    temporary = Path(tempfile.mkdtemp(prefix=".candidate_build_", dir=output_dir.parent))
    try:
        variant_manifests: dict[str, Any] = {}
        for base_variant, arm_plans in plans.items():
            arm_manifests: dict[str, Any] = {}
            for arm, plan in arm_plans.items():
                solver_name = f"{base_variant}__{arm}__candidate_solver.jsonl"
                decisions_name = f"{base_variant}__{arm}__decisions.jsonl"
                solver_path = temporary / solver_name
                with solver_path.open("wb") as handle:
                    for line in plan["output_lines"]:
                        handle.write(line)
                decisions_sha = _write_jsonl(
                    temporary / decisions_name, plan["decisions"]
                )
                solver_sha = sha256_file(solver_path)
                repairs = plan["counts"]["repair_from_raw_response"]
                source_exact = sum(
                    plan["output_lines"][i] == plan["base_raw"][i]
                    for i, task_id in enumerate(plan["order"])
                    if task_id in plan["protected"]
                )
                image_exact = sum(
                    plan["output_lines"][i] == plan["base_raw"][i]
                    for i, route in enumerate(plan["routes"])
                    if route == "image_judge"
                )
                exact_base_rows = sum(
                    output == base
                    for output, base in zip(
                        plan["output_lines"], plan["base_raw"], strict=True
                    )
                )
                arm_manifests[arm] = {
                    "label": (
                        "strict_development_arm"
                        if arm == "strict"
                        else "explicitly_post_score_motivated_exploratory_arm"
                    ),
                    "counts": {
                        "repaired_rows": repairs,
                        "base_passthrough_exact_bytes": exact_base_rows,
                        "eligible_deterministic_uncovered_invalid_answers": plan[
                            "invalid_eligible"
                        ],
                        "decision_reasons": dict(sorted(plan["counts"].items())),
                    },
                    "preservation": {
                        "source_union_rows": 156,
                        "source_union_exact_base_line_bytes": source_exact,
                        "image_judge_rows": 97,
                        "image_judge_exact_base_line_bytes": image_exact,
                    },
                    "artifacts": {
                        "candidate_solver": {
                            "path": solver_name,
                            "rows": ROWS,
                            "sha256": solver_sha,
                        },
                        "decisions": {
                            "path": decisions_name,
                            "rows": ROWS,
                            "sha256": decisions_sha,
                        },
                    },
                }
            variant_manifests[base_variant] = {
                "label": (
                    "post_score_development_on_audited_v1_2_primary_240_unscored"
                    if base_variant == "on_v1_2_primary_240"
                    else "post_score_development_on_final_9b_source_unscored"
                ),
                "base_solver": INPUTS[BASE_INPUT_KEYS[base_variant]],
                "arms": arm_manifests,
            }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "post_score_motivated_development_candidate_unscored_not_evaluated",
            "chronology": {
                "historical_residual_outcomes_known_before_design": True,
                "post_score_motivated": True,
                "blind_claim": False,
                "preregistered_claim": False,
                "rules_frozen_before_candidate_materialization": True,
            },
            "runtime_outcome_access": False,
            "model_closure": [MODEL],
            "supersedes": EXPECTED_PROFILE["supersedes"],
            "rule_freeze_sha256": rule_freeze_sha,
            "inputs": INPUTS,
            "rows": ROWS,
            "base_variants": variant_manifests,
        }
        manifest_sha = _write_json(temporary / "candidate_manifest.json", manifest)
        output_freeze = {
            "schema_version": OUTPUT_FREEZE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "development_candidate_frozen_unscored_not_evaluated",
            "rule_freeze_sha256": rule_freeze_sha,
            "runtime_outcome_access": False,
            "supersedes": EXPECTED_PROFILE["supersedes"],
            "candidate_manifest": {
                "path": "candidate_manifest.json",
                "sha256": manifest_sha,
            },
            "base_variants": {
                base_variant: {
                    "base_solver": variant_manifest["base_solver"],
                    "arms": {
                        arm: {"artifacts": arm_manifest["artifacts"]}
                        for arm, arm_manifest in variant_manifest["arms"].items()
                    },
                }
                for base_variant, variant_manifest in variant_manifests.items()
            },
        }
        _write_json(temporary / "CANDIDATE_OUTPUT_FREEZE.json", output_freeze)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_output(output_dir)


def verify_output(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    freeze_report = verify_rule_freeze()
    rule_freeze_sha = freeze_report["sha256"]
    output_freeze = _read_json(output_dir / "CANDIDATE_OUTPUT_FREEZE.json", "output freeze")
    manifest = _read_json(output_dir / "candidate_manifest.json", "candidate manifest")
    if output_freeze.get("schema_version") != OUTPUT_FREEZE_SCHEMA:
        raise RepairError("output freeze schema mismatch")
    if output_freeze.get("status") != "development_candidate_frozen_unscored_not_evaluated":
        raise RepairError("output freeze status mismatch")
    if output_freeze.get("rule_freeze_sha256") != rule_freeze_sha:
        raise RepairError("output freeze rule pin mismatch")
    if output_freeze.get("runtime_outcome_access") is not False:
        raise RepairError("output freeze runtime outcome access is not false")
    if output_freeze.get("supersedes") != EXPECTED_PROFILE["supersedes"]:
        raise RepairError("output freeze predecessor supersession mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RepairError("candidate manifest schema mismatch")
    if manifest.get("runtime_outcome_access") is not False or manifest.get("model_closure") != [MODEL]:
        raise RepairError("candidate manifest closure mismatch")
    if manifest.get("supersedes") != EXPECTED_PROFILE["supersedes"]:
        raise RepairError("candidate manifest predecessor supersession mismatch")
    if manifest.get("rule_freeze_sha256") != rule_freeze_sha:
        raise RepairError("candidate manifest rule pin mismatch")
    if output_freeze.get("candidate_manifest") != {
        "path": "candidate_manifest.json",
        "sha256": sha256_file(output_dir / "candidate_manifest.json"),
    }:
        raise RepairError("candidate manifest descriptor mismatch")
    if set(manifest.get("base_variants", {})) != set(BASE_VARIANTS):
        raise RepairError("candidate manifest base variant closure mismatch")
    if set(output_freeze.get("base_variants", {})) != set(BASE_VARIANTS):
        raise RepairError("output freeze base variant closure mismatch")
    repair_counts: dict[str, dict[str, int]] = {}
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    expected_preservation = {
        "source_union_rows": 156,
        "source_union_exact_base_line_bytes": 156,
        "image_judge_rows": 97,
        "image_judge_exact_base_line_bytes": 97,
    }
    for base_variant in BASE_VARIANTS:
        variant_manifest = manifest["base_variants"][base_variant]
        expected_base = INPUTS[BASE_INPUT_KEYS[base_variant]]
        if variant_manifest.get("base_solver") != expected_base:
            raise RepairError(f"base solver descriptor mismatch: {base_variant}")
        frozen_variant = output_freeze["base_variants"][base_variant]
        if frozen_variant.get("base_solver") != expected_base:
            raise RepairError(f"output freeze base solver mismatch: {base_variant}")
        if set(variant_manifest.get("arms", {})) != set(ARMS):
            raise RepairError(f"candidate arm closure mismatch: {base_variant}")
        if set(frozen_variant.get("arms", {})) != set(ARMS):
            raise RepairError(f"output freeze arm closure mismatch: {base_variant}")
        repair_counts[base_variant] = {}
        artifact_hashes[base_variant] = {}
        for arm in ARMS:
            arm_manifest = variant_manifest["arms"][arm]
            artifacts = arm_manifest.get("artifacts", {})
            if frozen_variant["arms"][arm] != {"artifacts": artifacts}:
                raise RepairError(f"output freeze artifact mismatch: {base_variant}/{arm}")
            if set(artifacts) != {"candidate_solver", "decisions"}:
                raise RepairError(f"candidate artifact closure mismatch: {base_variant}/{arm}")
            for label, descriptor in artifacts.items():
                path = output_dir / descriptor["path"]
                if descriptor.get("rows") != ROWS or sha256_file(path) != descriptor.get("sha256"):
                    raise RepairError(f"candidate artifact mismatch: {base_variant}/{arm}/{label}")
            solver_descriptor = artifacts["candidate_solver"]
            decisions_descriptor = artifacts["decisions"]
            solver_rows, solver_raw = _read_jsonl_raw(
                output_dir / solver_descriptor["path"],
                f"{base_variant}/{arm} candidate solver",
            )
            decision_rows, _ = _read_jsonl_raw(
                output_dir / decisions_descriptor["path"],
                f"{base_variant}/{arm} candidate decisions",
            )
            plan = _plan_candidate(rule_freeze_sha, arm, base_variant)
            if solver_raw != plan["output_lines"] or decision_rows != plan["decisions"]:
                raise RepairError(
                    f"{base_variant}/{arm} candidate is not deterministic replay"
                )
            if len(solver_rows) != ROWS or [row.get("task_id") for row in solver_rows] != plan["order"]:
                raise RepairError(f"{base_variant}/{arm} task closure mismatch")
            expected_repairs = plan["counts"]["repair_from_raw_response"]
            if arm_manifest.get("counts", {}).get("repaired_rows") != expected_repairs:
                raise RepairError(f"{base_variant}/{arm} repair count mismatch")
            if arm_manifest.get("preservation") != expected_preservation:
                raise RepairError(f"{base_variant}/{arm} byte preservation mismatch")
            repair_counts[base_variant][arm] = expected_repairs
            artifact_hashes[base_variant][arm] = {
                "candidate_solver_sha256": solver_descriptor["sha256"],
                "decisions_sha256": decisions_descriptor["sha256"],
            }
    return {
        "status": "development_candidate_verified_unscored_not_evaluated",
        "repairs": repair_counts,
        "rule_freeze_sha256": rule_freeze_sha,
        "base_variants": artifact_hashes,
        "candidate_manifest_sha256": sha256_file(output_dir / "candidate_manifest.json"),
        "output_freeze_sha256": sha256_file(output_dir / "CANDIDATE_OUTPUT_FREEZE.json"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-score-motivated, unscored 9B answer-contract repair development arm"
    )
    parser.add_argument("--write-rule-freeze", action="store_true")
    parser.add_argument("--verify-rule-freeze", action="store_true")
    parser.add_argument("--build-candidate", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    selected = sum(
        int(value)
        for value in (
            args.write_rule_freeze,
            args.verify_rule_freeze,
            args.build_candidate,
            args.verify_output,
        )
    )
    if selected != 1:
        raise RepairError("select exactly one action")
    if args.write_rule_freeze:
        report = write_rule_freeze()
    elif args.verify_rule_freeze:
        report = verify_rule_freeze()
    elif args.build_candidate:
        report = build_candidate(args.output_dir)
    else:
        report = verify_output(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

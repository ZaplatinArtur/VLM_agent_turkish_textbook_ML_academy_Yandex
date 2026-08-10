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
import unicodedata
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "maxim_9b_answer_canonicalization_v1"
MODEL = "Qwen/Qwen3.5-9B"
ROWS = 274
PROFILE_SCHEMA = "maxim-9b-answer-canonicalization-profile-v1"
PROJECTION_SCHEMA = "maxim-9b-observable-question-projection-v1"
PROJECTION_MANIFEST_SCHEMA = "maxim-9b-observable-question-projection-manifest-v1"
RULE_FREEZE_SCHEMA = "maxim-9b-answer-canonicalization-rule-freeze-v1"
DECISION_SCHEMA = "maxim-9b-answer-canonicalization-decision-v1"
ROW_PROVENANCE_SCHEMA = "maxim-9b-answer-canonicalization-row-provenance-v1"
MANIFEST_SCHEMA = "maxim-9b-answer-canonicalization-candidate-manifest-v1"
OUTPUT_FREEZE_SCHEMA = "maxim-9b-answer-canonicalization-output-freeze-v1"

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = EXPERIMENT_ROOT / "profile.json"
PROJECTION_PATH = EXPERIMENT_ROOT / "observable_question_projection.jsonl"
PROJECTION_MANIFEST_PATH = EXPERIMENT_ROOT / "OBSERVABLE_PROJECTION_MANIFEST.json"
RULE_FREEZE_PATH = EXPERIMENT_ROOT / "DEVELOPMENT_RULE_FREEZE.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "candidate_output"
EVALUATION_OUTPUT = EXPERIMENT_ROOT / "evaluation_output"

BENCHMARK = {
    "path": "artifacts/baselines/basic_page_rag_v1/validation_274.jsonl",
    "sha256": "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9",
}
INPUTS: dict[str, dict[str, str]] = {
    "base_solver": {
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

# Filled only after the outcome-free observable projection is materialized once.
OBSERVABLE_PROJECTION_SHA256 = "83636cdea696c70c069c562cfa6b7b20729331947cc534df79df205b2f3f43ff"
OBSERVABLE_PROJECTION_MANIFEST_SHA256 = "581860b18507ec4657c826a6fc0c83a7ff879495417cf2b87c45aeafa93532eb"

ARMS = (
    "choice_nfkc_only",
    "choice_curated_confusable_exploratory",
    "fraction_percent_explicit_question_contract_exploratory",
)
PROJECTION_FIELDS = ("row_index", "task_id", "question", "answer_type", "subject")
FORBIDDEN_RUNTIME_KEYS = frozenset(
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

# Deliberately small: these are single-codepoint glyph confusables for A/B/C/E.
# D is omitted because no similarly safe cross-script mapping exists.
CURATED_CHOICE_CONFUSABLES = {
    "Α": "A", "α": "A", "А": "A", "а": "A",
    "Β": "B", "β": "B", "В": "B", "в": "B",
    "Ϲ": "C", "ϲ": "C", "С": "C", "с": "C",
    "Ε": "E", "ε": "E", "Е": "E", "е": "E",
}

PERCENT_DEMAND_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byüzde\s+(?:olarak|biçiminde|şeklinde)\b",
        r"\bas\s+a\s+percentage\b",
        r"\bin\s+percent(?:age)?\b",
        r"\bpercentage\s+form\b",
        r"\bв\s+процентах\b",
        r"\bв\s+процентн(?:ом|ой)\s+виде\b",
        r"\bdalam\s+(?:bentuk\s+)?persen\b",
        r"\bfoiz\s+(?:ko['’]?rinishida|shaklida)\b",
        r"\bпайыз\s+(?:түрінде|күйінде)\b",
    )
)
FRACTION_DEMAND_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bkesir\s+(?:olarak|biçiminde|şeklinde)\b",
        r"\bsadeleştirilmiş\s+kesir\b",
        r"\bas\s+a\s+fraction\b",
        r"\bfraction\s+form\b",
        r"\bsimplest\s+fraction\b",
        r"\bв\s+виде\s+(?:обыкновенной\s+)?дроби\b",
        r"\bdalam\s+bentuk\s+pecahan\b",
        r"\bsebagai\s+pecahan\b",
        r"\bkasr\s+(?:ko['’]?rinishida|shaklida)\b",
        r"\bбөлшек\s+түрінде\b",
    )
)
FRACTION_RE = re.compile(r"\A([+-]?\d{1,12})\s*/\s*([+-]?\d{1,12})\Z")
PERCENT_RE = re.compile(r"\A([+-]?(?:\d{1,12}(?:[.,]\d{1,6})?|[.,]\d{1,6}))\s*%\Z")


class CanonicalizationError(RuntimeError):
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
        raise CanonicalizationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CanonicalizationError(f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalizationError(f"{label} invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalizationError(f"{label} must be object")
    return value


def _read_jsonl_raw(path: Path, label: str) -> tuple[list[dict[str, Any]], list[bytes]]:
    try:
        lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
    except OSError as exc:
        raise CanonicalizationError(f"cannot read {label}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines):
        if not raw.endswith(b"\n"):
            raise CanonicalizationError(f"{label}:{index} lacks terminating newline")
        try:
            row = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CanonicalizationError(f"{label}:{index} invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CanonicalizationError(f"{label}:{index} must be object")
        rows.append(row)
    return rows, lines


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk(item)


def _assert_runtime_clean(value: Any, label: str) -> None:
    for key, item in _walk(value):
        lowered = key.casefold() if key is not None else ""
        if lowered in FORBIDDEN_RUNTIME_KEYS:
            raise CanonicalizationError(f"{label}: forbidden runtime field {key}")
        if lowered == "gold_access" and item is not False:
            raise CanonicalizationError(f"{label}: non-false gold_access")
        if lowered == "model" and item not in (None, MODEL):
            raise CanonicalizationError(f"{label}: model closure violation {item!r}")
        if lowered == "inherited_27b_outputs" and item is not False:
            raise CanonicalizationError(f"{label}: inherited 27B output")


def _safe_repo_path(descriptor: Mapping[str, str], label: str) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CanonicalizationError(f"{label}: unsafe path")
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise CanonicalizationError(f"{label}: path escaped repository") from exc
    if sha256_file(path) != descriptor["sha256"]:
        raise CanonicalizationError(f"{label}: SHA-256 mismatch")
    return path


def build_observable_projection() -> dict[str, Any]:
    if PROJECTION_PATH.exists() or PROJECTION_MANIFEST_PATH.exists():
        raise CanonicalizationError("observable projection already exists; refusing overwrite")
    benchmark_path = _safe_repo_path(BENCHMARK, "benchmark design input")
    benchmark_rows, _ = _read_jsonl_raw(benchmark_path, "benchmark design input")
    if len(benchmark_rows) != ROWS:
        raise CanonicalizationError("benchmark design input is not full274")
    projection: list[dict[str, Any]] = []
    for index, source in enumerate(benchmark_rows):
        task_id = source.get("task_id")
        question = source.get("question")
        answer_type = source.get("answer_type")
        subject = source.get("subject")
        if not all(isinstance(value, str) for value in (task_id, question, answer_type, subject)):
            raise CanonicalizationError(f"benchmark observable fields malformed at row {index}")
        row = {
            "schema_version": PROJECTION_SCHEMA,
            "row_index": index,
            "task_id": task_id,
            "question": question,
            "answer_type": answer_type,
            "subject": subject,
        }
        _assert_runtime_clean(row, f"projection row {index}")
        projection.append(row)
    if len({row["task_id"] for row in projection}) != ROWS:
        raise CanonicalizationError("projection task identities are not unique")
    projection_bytes = b"".join(canonical_json(row) for row in projection)
    projection_sha = sha256_bytes(projection_bytes)
    manifest = {
        "schema_version": PROJECTION_MANIFEST_SCHEMA,
        "status": "post_score_design_observable_projection_frozen_no_outcomes",
        "historical_outcomes_known_before_projection": True,
        "runtime_outcome_access": False,
        "source_benchmark_sha256": BENCHMARK["sha256"],
        "rows": ROWS,
        "allowlisted_fields": ["ordered_identity", "question", "answer_type", "subject"],
        "excluded_fields": [
            "reference_answer",
            "reference_solution",
            "gold",
            "correctness",
            "score",
            "judge",
            "outcome",
            "question_images",
            "grade",
        ],
        "projection": {
            "path": PROJECTION_PATH.name,
            "sha256": projection_sha,
            "rows": ROWS,
        },
    }
    temporary_projection = PROJECTION_PATH.with_suffix(".tmp")
    temporary_manifest = PROJECTION_MANIFEST_PATH.with_suffix(".tmp")
    temporary_projection.write_bytes(projection_bytes)
    temporary_manifest.write_bytes(canonical_json(manifest))
    os.replace(temporary_projection, PROJECTION_PATH)
    os.replace(temporary_manifest, PROJECTION_MANIFEST_PATH)
    return {
        "status": manifest["status"],
        "projection_sha256": projection_sha,
        "manifest_sha256": sha256_file(PROJECTION_MANIFEST_PATH),
        "rows": ROWS,
    }


def _valid_present_top_level_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    if text[0] in '{["':
        return None
    return text


def normalize_choice_nfkc(answer: str, answer_type: str) -> str | None:
    if answer_type != "choice":
        return None
    text = _valid_present_top_level_answer(answer)
    if text is None or len(text) != 1 or text.upper() in "ABCDE":
        return None
    normalized = unicodedata.normalize("NFKC", text)
    if normalized == text or len(normalized) != 1:
        return None
    choice = normalized.upper()
    return choice if choice in "ABCDE" else None


def normalize_choice_curated(answer: str, answer_type: str) -> str | None:
    nfkc = normalize_choice_nfkc(answer, answer_type)
    if nfkc is not None:
        return nfkc
    if answer_type != "choice":
        return None
    text = _valid_present_top_level_answer(answer)
    if text is None or len(text) != 1 or text.upper() in "ABCDE":
        return None
    return CURATED_CHOICE_CONFUSABLES.get(text)


def _question_contract(question: str) -> str | None:
    percent = any(pattern.search(question) for pattern in PERCENT_DEMAND_PATTERNS)
    fraction = any(pattern.search(question) for pattern in FRACTION_DEMAND_PATTERNS)
    if percent == fraction:
        return None
    return "percent" if percent else "fraction"


def _fraction_to_percent(text: str) -> str | None:
    match = FRACTION_RE.fullmatch(text)
    if not match:
        return None
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if denominator == 0:
        return None
    value = Fraction(numerator * 100, denominator)
    reduced_denominator = value.denominator
    for prime in (2, 5):
        while reduced_denominator % prime == 0:
            reduced_denominator //= prime
    if reduced_denominator != 1:
        return None
    scale = 0
    power = 1
    while power % value.denominator != 0 and scale <= 6:
        power *= 10
        scale += 1
    if scale > 6:
        return None
    scaled = value.numerator * power // value.denominator
    if scale == 0:
        rendered = str(scaled)
    else:
        sign = "-" if scaled < 0 else ""
        digits = str(abs(scaled)).zfill(scale + 1)
        rendered = f"{sign}{digits[:-scale]}.{digits[-scale:]}".rstrip("0").rstrip(".")
    return f"{rendered}%"


def _percent_to_fraction(text: str) -> str | None:
    match = PERCENT_RE.fullmatch(text)
    if not match:
        return None
    number = match.group(1).replace(",", ".")
    sign = -1 if number.startswith("-") else 1
    unsigned = number.lstrip("+-")
    if "." in unsigned:
        whole, decimals = unsigned.split(".", 1)
    else:
        whole, decimals = unsigned, ""
    numerator = sign * int((whole or "0") + decimals)
    denominator = (10 ** len(decimals)) * 100
    value = Fraction(numerator, denominator)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def normalize_fraction_percent(answer: str, answer_type: str, question: str) -> str | None:
    if answer_type != "short_text":
        return None
    text = _valid_present_top_level_answer(answer)
    if text is None or len(text) > 64:
        return None
    contract = _question_contract(question)
    if contract == "percent":
        return _fraction_to_percent(text)
    if contract == "fraction":
        return _percent_to_fraction(text)
    return None


def decide_row(
    row: Mapping[str, Any],
    observable: Mapping[str, Any],
    *,
    evaluation_route: str,
    protected_by_source_union: bool,
    arm: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise CanonicalizationError(f"unknown arm {arm!r}")
    if protected_by_source_union:
        return {"action": "preserve_base_exact_bytes", "reason": "protected_by_source_union"}
    if evaluation_route == "image_judge":
        return {"action": "preserve_base_exact_bytes", "reason": "image_judge_route"}
    if evaluation_route != "deterministic":
        raise CanonicalizationError(f"unknown route {evaluation_route!r}")
    answer = row.get("final_answer")
    if _valid_present_top_level_answer(answer) is None:
        return {
            "action": "preserve_base_exact_bytes",
            "reason": "invalid_or_absent_answer_owned_by_separate_explicit_json_successor",
        }
    answer_type = observable["answer_type"]
    if arm == "choice_nfkc_only":
        candidate = normalize_choice_nfkc(answer, answer_type)
        reason = "nfkc_single_choice_canonicalization"
    elif arm == "choice_curated_confusable_exploratory":
        candidate = normalize_choice_curated(answer, answer_type)
        reason = "curated_single_choice_confusable_canonicalization"
    else:
        candidate = normalize_fraction_percent(answer, answer_type, observable["question"])
        reason = "explicit_question_contract_fraction_percent_canonicalization"
    if candidate is None or candidate == answer.strip():
        return {"action": "preserve_base_exact_bytes", "reason": "no_generic_canonicalization"}
    return {"action": "canonicalize_top_level_final_answer", "reason": reason, "candidate": candidate}


def _load_projection() -> tuple[list[dict[str, Any]], list[bytes]]:
    if sha256_file(PROJECTION_PATH) != OBSERVABLE_PROJECTION_SHA256:
        raise CanonicalizationError("observable projection SHA mismatch")
    if sha256_file(PROJECTION_MANIFEST_PATH) != OBSERVABLE_PROJECTION_MANIFEST_SHA256:
        raise CanonicalizationError("observable projection manifest SHA mismatch")
    manifest = _read_json(PROJECTION_MANIFEST_PATH, "observable projection manifest")
    expected_manifest = {
        "schema_version": PROJECTION_MANIFEST_SCHEMA,
        "status": "post_score_design_observable_projection_frozen_no_outcomes",
        "historical_outcomes_known_before_projection": True,
        "runtime_outcome_access": False,
        "source_benchmark_sha256": BENCHMARK["sha256"],
        "rows": ROWS,
        "allowlisted_fields": ["ordered_identity", "question", "answer_type", "subject"],
        "excluded_fields": [
            "reference_answer", "reference_solution", "gold", "correctness", "score",
            "judge", "outcome", "question_images", "grade",
        ],
        "projection": {
            "path": PROJECTION_PATH.name,
            "sha256": OBSERVABLE_PROJECTION_SHA256,
            "rows": ROWS,
        },
    }
    if manifest != expected_manifest:
        raise CanonicalizationError("observable projection manifest contract mismatch")
    rows, raw = _read_jsonl_raw(PROJECTION_PATH, "observable projection")
    if len(rows) != ROWS:
        raise CanonicalizationError("observable projection row count mismatch")
    for index, row in enumerate(rows):
        if set(row) != {"schema_version", *PROJECTION_FIELDS}:
            raise CanonicalizationError(f"projection row {index} schema mismatch")
        if row["schema_version"] != PROJECTION_SCHEMA or row["row_index"] != index:
            raise CanonicalizationError(f"projection row {index} order mismatch")
        if not all(isinstance(row[field], str) for field in ("task_id", "question", "answer_type", "subject")):
            raise CanonicalizationError(f"projection row {index} types mismatch")
        _assert_runtime_clean(row, f"projection row {index}")
    return rows, raw


def _load_authorities() -> dict[str, Any]:
    paths = {name: _safe_repo_path(value, name) for name, value in INPUTS.items()}
    base_rows, base_raw = _read_jsonl_raw(paths["base_solver"], "base solver")
    order = _read_json(paths["benchmark_order"], "benchmark order").get("rows")
    route_rows = _read_json(paths["route_map"], "route map").get("rows")
    protected_rows = _read_json(paths["source_union_membership"], "source union").get("task_ids")
    projection, _ = _load_projection()
    if not isinstance(order, list) or len(order) != ROWS or len(set(order)) != ROWS:
        raise CanonicalizationError("benchmark order is not 274 unique IDs")
    if not isinstance(route_rows, list) or len(route_rows) != ROWS:
        raise CanonicalizationError("route map is not full274")
    if not isinstance(protected_rows, list) or len(protected_rows) != 156 or len(set(protected_rows)) != 156:
        raise CanonicalizationError("source union is not 156 unique IDs")
    if len(base_rows) != ROWS or [row.get("task_id") for row in base_rows] != order:
        raise CanonicalizationError("base solver identity/order mismatch")
    if [row["task_id"] for row in projection] != order:
        raise CanonicalizationError("observable projection identity/order mismatch")
    routes: list[str] = []
    for index, (task_id, route_row) in enumerate(zip(order, route_rows, strict=True)):
        expected = {"row_index": index, "task_id": task_id, "evaluation_route": route_row.get("evaluation_route")}
        if route_row != expected or route_row.get("evaluation_route") not in {"deterministic", "image_judge"}:
            raise CanonicalizationError(f"route authority mismatch at row {index}")
        routes.append(route_row["evaluation_route"])
    if Counter(routes) != {"deterministic": 177, "image_judge": 97}:
        raise CanonicalizationError("route split changed")
    for index, row in enumerate(base_rows):
        if row.get("model") != MODEL:
            raise CanonicalizationError(f"base row {index}: model closure mismatch")
        generation = row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise CanonicalizationError(f"base row {index}: gold_access not false")
        _assert_runtime_clean(row, f"base row {index}")
    return {
        "base_rows": base_rows,
        "base_raw": base_raw,
        "order": order,
        "routes": routes,
        "protected": frozenset(protected_rows),
        "projection": projection,
    }


def _expected_profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "post_score_motivated_development_not_blind_not_preregistered",
        "chronology": {
            "historical_240_score_and_task_outcomes_known_before_design": True,
            "post_score_motivated": True,
            "blind_claim": False,
            "preregistered_claim": False,
            "rules_frozen_before_candidate_materialization_and_new_arm_evaluation": True,
            "candidate_must_remain_unscored_and_unevaluated": True,
        },
        "model_closure": [MODEL],
        "runtime_outcome_access": False,
        "inputs": {
            **INPUTS,
            "observable_projection": {"path": PROJECTION_PATH.name, "sha256": OBSERVABLE_PROJECTION_SHA256},
            "observable_projection_manifest": {
                "path": PROJECTION_MANIFEST_PATH.name,
                "sha256": OBSERVABLE_PROJECTION_MANIFEST_SHA256,
            },
        },
        "scope": {
            "eligible_route": "deterministic",
            "eligible_membership": "outside_pinned_source_union",
            "preserve_source_union_156_as_exact_base_line_bytes": True,
            "preserve_image_judge_97_as_exact_base_line_bytes": True,
            "task_id_lists_or_rules": False,
            "scorer_or_gold_edits": False,
        },
        "division_of_responsibility": {
            "explicit_json_recovery_owned_by": "maxim_9b_answer_contract_repair_v1_1",
            "this_experiment_reads_or_parses_raw_response": False,
            "invalid_or_absent_top_level_answers_are_never_changed_here": True,
        },
        "arms": {
            "choice_nfkc_only": {
                "role": "primary_conservative",
                "rule": "answer_type_choice_and_single_codepoint_NFKC_maps_exactly_to_ASCII_A_E",
            },
            "choice_curated_confusable_exploratory": {
                "role": "post_score_exploratory",
                "rule": "primary_rule_or_single_whole_answer_codepoint_in_frozen_Greek_Cyrillic_A_B_C_E_map",
            },
            "fraction_percent_explicit_question_contract_exploratory": {
                "role": "post_score_exploratory",
                "rule": "short_text_only_and_observable_question_contains_frozen_multilingual_explicit_output_form_phrase",
                "bare_percent_symbol_is_not_a_contract_signal": True,
                "placeholder_or_image_only_question_never_triggers": True,
                "fraction_to_percent_requires_exact_terminating_decimal_at_most_6_places": True,
            },
        },
        "mutation": {
            "copy_full_base_row": True,
            "replace_only_top_level_final_answer": True,
            "retain_raw_response_byte_content": True,
            "add_explicit_generation_provenance": True,
            "fail_closed": True,
        },
    }


def _profile() -> dict[str, Any]:
    profile = _read_json(PROFILE_PATH, "profile")
    if profile != _expected_profile():
        raise CanonicalizationError("profile differs from exact code contract")
    return profile


def _rule_freeze_payload() -> dict[str, Any]:
    return {
        "schema_version": RULE_FREEZE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "development_rules_frozen_before_candidate_build_and_new_arm_evaluation",
        "chronology": {
            "historical_240_score_and_task_outcomes_known_before_design": True,
            "post_score_motivated": True,
            "blind_claim": False,
            "preregistered_claim": False,
            "candidate_output_absent_at_freeze": True,
            "new_arm_scores_absent_at_freeze": True,
        },
        "runtime_outcome_access": False,
        "inputs": _expected_profile()["inputs"],
        "artifacts": {
            "profile": {"path": PROFILE_PATH.name, "sha256": sha256_file(PROFILE_PATH)},
            "code": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__).resolve())},
            "tests": {
                "path": "test_answer_canonicalization_v1.py",
                "sha256": sha256_file(EXPERIMENT_ROOT / "test_answer_canonicalization_v1.py"),
            },
            "readme": {"path": "README.md", "sha256": sha256_file(EXPERIMENT_ROOT / "README.md")},
        },
    }


def write_rule_freeze() -> dict[str, Any]:
    _profile()
    _load_authorities()
    if RULE_FREEZE_PATH.exists() or DEFAULT_OUTPUT.exists() or EVALUATION_OUTPUT.exists():
        raise CanonicalizationError("rule freeze, candidate output, or evaluation output already exists")
    temporary = RULE_FREEZE_PATH.with_suffix(".tmp")
    temporary.write_bytes(canonical_json(_rule_freeze_payload()))
    os.replace(temporary, RULE_FREEZE_PATH)
    return verify_rule_freeze()


def verify_rule_freeze() -> dict[str, Any]:
    _profile()
    _load_authorities()
    freeze = _read_json(RULE_FREEZE_PATH, "rule freeze")
    if freeze != _rule_freeze_payload():
        raise CanonicalizationError("rule freeze payload/hash closure mismatch")
    return {"status": "development_rule_freeze_verified", "sha256": sha256_file(RULE_FREEZE_PATH)}


def _plan_candidate(rule_freeze_sha: str, arm: str) -> dict[str, Any]:
    bound = _load_authorities()
    output_lines: list[bytes] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, task_id in enumerate(bound["order"]):
        base = bound["base_rows"][index]
        base_raw = bound["base_raw"][index]
        route = bound["routes"][index]
        protected = task_id in bound["protected"]
        observable = bound["projection"][index]
        decision = decide_row(
            base,
            observable,
            evaluation_route=route,
            protected_by_source_union=protected,
            arm=arm,
        )
        if decision["action"] == "canonicalize_top_level_final_answer":
            changed = copy.deepcopy(base)
            original_answer = changed["final_answer"]
            changed["final_answer"] = decision["candidate"]
            generation = changed.get("generation")
            if not isinstance(generation, dict):
                raise CanonicalizationError(f"row {index}: generation missing")
            generation["answer_canonicalization_v1"] = {
                "schema_version": ROW_PROVENANCE_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "arm": arm,
                "rule_freeze_sha256": rule_freeze_sha,
                "action": decision["reason"],
                "base_row_sha256": sha256_bytes(base_raw),
                "original_answer_sha256": sha256_bytes(original_answer.encode("utf-8")),
                "canonical_answer_sha256": sha256_bytes(decision["candidate"].encode("utf-8")),
                "observable_question_row_sha256": sha256_bytes(canonical_json(observable)),
                "authoritative_route": route,
                "protected_by_source_union": False,
                "runtime_outcome_access": False,
            }
            _assert_runtime_clean(changed, f"candidate row {index}")
            output_raw = canonical_json(changed)
            counts["canonicalized_rows"] += 1
        else:
            output_raw = base_raw
            counts[decision["reason"]] += 1
        output_lines.append(output_raw)
        decisions.append(
            {
                "schema_version": DECISION_SCHEMA,
                "arm": arm,
                "row_index": index,
                "task_id": task_id,
                "authoritative_route": route,
                "protected_by_source_union": protected,
                "action": decision["action"],
                "reason": decision["reason"],
                "base_row_sha256": sha256_bytes(base_raw),
                "output_row_sha256": sha256_bytes(output_raw),
                "runtime_outcome_access": False,
            }
        )
    return {**bound, "output_lines": output_lines, "decisions": decisions, "counts": counts}


def build_candidates(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists():
        raise CanonicalizationError("candidate output exists; refusing overwrite")
    freeze = verify_rule_freeze()
    rule_sha = freeze["sha256"]
    plans = {arm: _plan_candidate(rule_sha, arm) for arm in ARMS}
    temporary = Path(tempfile.mkdtemp(prefix=".canonicalization_", dir=output_dir.parent))
    try:
        arms: dict[str, Any] = {}
        for arm, plan in plans.items():
            solver_name = f"{arm}_candidate_solver.jsonl"
            decisions_name = f"{arm}_decisions.jsonl"
            solver_path = temporary / solver_name
            decisions_path = temporary / decisions_name
            solver_path.write_bytes(b"".join(plan["output_lines"]))
            decisions_path.write_bytes(b"".join(canonical_json(row) for row in plan["decisions"]))
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
            arms[arm] = {
                "canonicalized_rows": plan["counts"]["canonicalized_rows"],
                "decision_reasons": dict(sorted(plan["counts"].items())),
                "preservation": {
                    "source_union_rows": 156,
                    "source_union_exact_base_line_bytes": source_exact,
                    "image_judge_rows": 97,
                    "image_judge_exact_base_line_bytes": image_exact,
                },
                "artifacts": {
                    "candidate_solver": {
                        "path": solver_name, "rows": ROWS, "sha256": sha256_file(solver_path)
                    },
                    "decisions": {
                        "path": decisions_name, "rows": ROWS, "sha256": sha256_file(decisions_path)
                    },
                },
            }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "post_score_development_candidates_frozen_unscored_not_evaluated",
            "historical_240_score_known_before_design": True,
            "blind_claim": False,
            "preregistered_claim": False,
            "runtime_outcome_access": False,
            "model_closure": [MODEL],
            "rule_freeze_sha256": rule_sha,
            "rows": ROWS,
            "arms": arms,
        }
        manifest_path = temporary / "candidate_manifest.json"
        manifest_path.write_bytes(canonical_json(manifest))
        output_freeze = {
            "schema_version": OUTPUT_FREEZE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "candidate_outputs_frozen_unscored_not_evaluated",
            "runtime_outcome_access": False,
            "rule_freeze_sha256": rule_sha,
            "candidate_manifest": {"path": manifest_path.name, "sha256": sha256_file(manifest_path)},
            "arms": {arm: value["artifacts"] for arm, value in arms.items()},
        }
        (temporary / "CANDIDATE_OUTPUT_FREEZE.json").write_bytes(canonical_json(output_freeze))
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_output(output_dir)


def verify_output(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rule_sha = verify_rule_freeze()["sha256"]
    manifest = _read_json(output_dir / "candidate_manifest.json", "candidate manifest")
    output_freeze = _read_json(output_dir / "CANDIDATE_OUTPUT_FREEZE.json", "output freeze")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("status") != "post_score_development_candidates_frozen_unscored_not_evaluated"
        or manifest.get("historical_240_score_known_before_design") is not True
        or manifest.get("blind_claim") is not False
        or manifest.get("preregistered_claim") is not False
        or manifest.get("rows") != ROWS
        or manifest.get("rule_freeze_sha256") != rule_sha
    ):
        raise CanonicalizationError("candidate manifest contract mismatch")
    if manifest.get("runtime_outcome_access") is not False or manifest.get("model_closure") != [MODEL]:
        raise CanonicalizationError("candidate manifest runtime/model closure mismatch")
    if (
        output_freeze.get("schema_version") != OUTPUT_FREEZE_SCHEMA
        or output_freeze.get("experiment_id") != EXPERIMENT_ID
        or output_freeze.get("status") != "candidate_outputs_frozen_unscored_not_evaluated"
        or output_freeze.get("runtime_outcome_access") is not False
        or output_freeze.get("rule_freeze_sha256") != rule_sha
    ):
        raise CanonicalizationError("candidate output freeze contract mismatch")
    if output_freeze.get("candidate_manifest") != {
        "path": "candidate_manifest.json",
        "sha256": sha256_file(output_dir / "candidate_manifest.json"),
    }:
        raise CanonicalizationError("candidate manifest hash descriptor mismatch")
    if set(manifest.get("arms", {})) != set(ARMS) or set(output_freeze.get("arms", {})) != set(ARMS):
        raise CanonicalizationError("candidate arm closure mismatch")
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for arm in ARMS:
        plan = _plan_candidate(rule_sha, arm)
        arm_manifest = manifest["arms"][arm]
        artifacts = arm_manifest["artifacts"]
        if output_freeze["arms"][arm] != artifacts:
            raise CanonicalizationError(f"{arm}: output freeze artifact mismatch")
        solver_path = output_dir / artifacts["candidate_solver"]["path"]
        decisions_path = output_dir / artifacts["decisions"]["path"]
        solver_rows, solver_raw = _read_jsonl_raw(solver_path, f"{arm} solver")
        decision_rows, _ = _read_jsonl_raw(decisions_path, f"{arm} decisions")
        if (
            artifacts["candidate_solver"].get("rows") != ROWS
            or artifacts["decisions"].get("rows") != ROWS
        ):
            raise CanonicalizationError(f"{arm}: artifact row declaration mismatch")
        if sha256_file(solver_path) != artifacts["candidate_solver"]["sha256"]:
            raise CanonicalizationError(f"{arm}: solver SHA mismatch")
        if sha256_file(decisions_path) != artifacts["decisions"]["sha256"]:
            raise CanonicalizationError(f"{arm}: decisions SHA mismatch")
        if solver_raw != plan["output_lines"] or decision_rows != plan["decisions"]:
            raise CanonicalizationError(f"{arm}: output is not deterministic replay")
        if len(solver_rows) != ROWS or [row.get("task_id") for row in solver_rows] != plan["order"]:
            raise CanonicalizationError(f"{arm}: task closure mismatch")
        if arm_manifest["preservation"] != {
            "source_union_rows": 156,
            "source_union_exact_base_line_bytes": 156,
            "image_judge_rows": 97,
            "image_judge_exact_base_line_bytes": 97,
        }:
            raise CanonicalizationError(f"{arm}: protected preservation mismatch")
        expected_count = plan["counts"]["canonicalized_rows"]
        if arm_manifest["canonicalized_rows"] != expected_count:
            raise CanonicalizationError(f"{arm}: candidate count mismatch")
        if arm_manifest["decision_reasons"] != dict(sorted(plan["counts"].items())):
            raise CanonicalizationError(f"{arm}: decision reason counts mismatch")
        counts[arm] = expected_count
        hashes[arm] = artifacts["candidate_solver"]["sha256"]
    return {
        "status": "candidate_outputs_verified_unscored_not_evaluated",
        "counts": counts,
        "solver_sha256": hashes,
        "rule_freeze_sha256": rule_sha,
        "candidate_manifest_sha256": sha256_file(output_dir / "candidate_manifest.json"),
        "output_freeze_sha256": sha256_file(output_dir / "CANDIDATE_OUTPUT_FREEZE.json"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-score generic answer canonicalization arms")
    parser.add_argument("--build-observable-projection", action="store_true")
    parser.add_argument("--write-rule-freeze", action="store_true")
    parser.add_argument("--verify-rule-freeze", action="store_true")
    parser.add_argument("--build-candidates", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    actions = [
        args.build_observable_projection,
        args.write_rule_freeze,
        args.verify_rule_freeze,
        args.build_candidates,
        args.verify_output,
    ]
    if sum(map(int, actions)) != 1:
        raise CanonicalizationError("select exactly one action")
    if args.build_observable_projection:
        report = build_observable_projection()
    elif args.write_rule_freeze:
        report = write_rule_freeze()
    elif args.verify_rule_freeze:
        report = verify_rule_freeze()
    elif args.build_candidates:
        report = build_candidates()
    else:
        report = verify_output()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-outcome, identity-free selective fusion for Maxim274.

The arbiter compares only the two proposed answers, public ``answer_type`` and
the generic row's frozen structural validity.  Task IDs are consumed solely by
the outer adapter after the decision, to align the selected answer with the
274-row output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
CANDIDATE = EXPERIMENTS / "maxim_9b_maxim274_hybrid_generic_siliconflow_nonstream_v1_3_20260812"
HYBRID = EXPERIMENTS / "maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812"
BASE251 = EXPERIMENTS / "maxim_9b_content_source_router_noid_v1_20260812" / "output/frozen/arms/strict_b_over_base240/solver.jsonl"
PUBLIC_QUEUE = EXPERIMENTS / "maxim_9b_maxim274_generic_content_adapter_v1_20260812" / "frozen/maxim274_public_runtime_queue.jsonl"
PREDICTIONS = CANDIDATE / "runs/generic_predictions_256.jsonl"
COMPLETION = CANDIDATE / "COMPLETION.json"
ALIGNMENT = CANDIDATE / "frozen/outer_alignment_256.jsonl"
GENERIC_QUEUE = CANDIDATE / "frozen/queue_content_only_256.jsonl"
DECISIONS = HYBRID / "runs/maxim274/route_decisions.jsonl"
NORMALIZATION = REPO / "src/vlm_judge/normalization.py"
OUTPUT = HERE / "runs/selective_fusion_solver_274.jsonl"
MANIFEST = HERE / "FUSION_COMPLETION.json"
FREEZE = HERE / "FUSION_RULE_FREEZE.json"
FREEZE_SIDECAR = HERE / "FUSION_RULE_FREEZE_SHA256.txt"
AUDIT = HERE / "INDEPENDENT_AUDIT.json"

CANDIDATE_FREEZE_SHA = "7096b70b25281da114d70956fd8f7a468372eb315c955584fe83217b30bc630d"
CANDIDATE_AUDIT_SHA = "943c17d8f4a9be0876b38c170ac554db751df49c0c243146795a1e8024445570"
BASE251_SHA = "f87f6ad41817c3d55fde5630781cd6f9f958350bfde72bdebeb8567b454c832a"
PUBLIC_QUEUE_SHA = "134281d4ba1d9828b686974d36fdaaa599c4b365907d9f97082d90863f982101"
GENERIC_QUEUE_SHA = "c8e16e97a32ebea4a1a43a4273ea0109d44ce1da5e067d32223872cd53dd0bb2"
ALIGNMENT_SHA = "205b627f8eddf7376068787e12b7f5ffda7cf13f610d9240cbc1077e99b8f0b8"
DECISIONS_SHA = "0f6a31ec8d862ba67f61a8d90e96a0bd1e2a77d071a51ff1a36c512eeaea974c"
NORMALIZATION_SHA = "3e62b6ae5f020e75718a0e2ca4e173fa2474c3154fa32741f6958828beacf7dd"
GENERIC_SCHEMA = {
    "schema_version", "task_id", "final_answer", "option_label",
    "answer_type", "input_mode", "error", "generation",
}
GENERIC_OBSERVABLE_SCHEMA = GENERIC_SCHEMA - {"task_id"} | {"outer_schema_exact"}
ANSWER_TYPES = {"choice", "numeric", "short_text", "free_form"}


class FusionError(RuntimeError):
    """A frozen closure, identity, or answer-contract invariant failed."""


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or len(data) != before.st_size:
        raise FusionError(f"file changed during read: {path}")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(stable_bytes(path)).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def descriptor(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.resolve().relative_to(HERE.resolve()).as_posix() if path.resolve().is_relative_to(HERE.resolve()) else os.path.relpath(path.resolve(), HERE.resolve()).replace("\\", "/"),
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(stable_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FusionError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise FusionError(f"JSON is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FusionError(f"invalid JSONL row {path}:{number}") from exc
        if type(value) is not dict:
            raise FusionError(f"non-object JSONL row {path}:{number}")
        rows.append(value)
    return rows


def exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor_fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor_fd, view)
            if written <= 0:
                raise FusionError("short output write")
            view = view[written:]
        os.fsync(descriptor_fd)
    finally:
        os.close(descriptor_fd)
    if stable_bytes(path) != data or sha256(path) != hashlib.sha256(data).hexdigest():
        raise FusionError("post-write byte/hash mismatch")


def _load_normalizer() -> Any:
    if sha256(NORMALIZATION) != NORMALIZATION_SHA:
        raise FusionError("normalizer pin mismatch")
    spec = importlib.util.spec_from_file_location("frozen_fusion_normalizer", NORMALIZATION)
    if spec is None or spec.loader is None:
        raise FusionError("cannot load frozen normalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_parseable(answer: Any, answer_type: str, normalizer: Any) -> bool:
    """Conservative syntax predicate; it never claims semantic correctness."""
    text = str(answer or "").strip()
    if not text or len(text) > 120 or answer_type not in ANSWER_TYPES:
        return False
    if answer_type == "choice":
        return normalizer.normalize_multiple_choice(text) is not None
    if answer_type == "numeric":
        return normalizer.parse_numeric(text) is not None
    return bool(normalizer.normalize_text(text))


def generic_projection(row: Any, expected_id: str, expected_answer_type: str) -> dict[str, Any]:
    """Validate outer alignment, then remove identity before arbitration."""
    if type(row) is not dict or row.get("task_id") != expected_id:
        raise FusionError("prediction/alignment identity mismatch")
    return {
        **{key: row.get(key) for key in GENERIC_SCHEMA if key != "task_id"},
        "outer_schema_exact": set(row) == GENERIC_SCHEMA,
    }


def generic_valid(row: Any, answer_type: str) -> bool:
    """Recheck the exact frozen generic content contract, without identity."""
    if type(row) is not dict or set(row) != GENERIC_OBSERVABLE_SCHEMA or row.get("outer_schema_exact") is not True:
        return False
    if row.get("schema_version") != "maxim256-hybrid-generic-prediction-v1" or row.get("answer_type") != answer_type or row.get("input_mode") != "ocr_only" or row.get("error") is not None:
        return False
    answer, option = row.get("final_answer"), row.get("option_label")
    if type(answer) is not str or not answer.strip() or len(answer) > 1000 or type(option) is not str:
        return False
    if (answer_type == "choice" and option not in frozenset("ABCDE")) or (answer_type != "choice" and option != "NA"):
        return False
    return row.get("generation") == {
        "gold_access": False,
        "outcome_access": False,
        "model": "qwen/qwen3.5-9b",
        "provider": "SiliconFlow",
        "quantization": "fp8",
    }


def official_generic_answer(row: Mapping[str, Any], answer_type: str) -> str:
    return str(row["option_label"] if answer_type == "choice" else row["final_answer"]).strip()


def select_observable(*, answer_type: str, baseline_answer: Any, generic_projection: Any, normalizer: Any) -> dict[str, Any]:
    """Choose using only answer content/type and structural validity.

    The valid-but-wrong generic risk is deliberately bounded: a semantically
    unverified generic answer can replace the DB/base candidate only when the
    latter is not even parseable under the public answer contract.
    """
    baseline_ok = baseline_parseable(baseline_answer, answer_type, normalizer)
    generic_ok = generic_valid(generic_projection, answer_type)
    if not generic_ok:
        return {"selected": "baseline", "reason": "generic_invalid_fail_closed", "baseline_parseable": baseline_ok, "generic_valid": False}
    if baseline_ok:
        return {"selected": "baseline", "reason": "baseline_parseable_conservative_hold", "baseline_parseable": True, "generic_valid": True}
    return {"selected": "generic", "reason": "valid_generic_replaces_unparseable_baseline", "baseline_parseable": False, "generic_valid": True}


def _verify_descriptor(descriptor_value: Mapping[str, Any]) -> None:
    if set(descriptor_value) - {"path", "sha256", "size", "rows"}:
        raise FusionError("unexpected descriptor key")
    path = (HERE / str(descriptor_value["path"])).resolve()
    if not path.is_file() or path.stat().st_size != descriptor_value.get("size") or sha256(path) != descriptor_value.get("sha256"):
        raise FusionError(f"frozen descriptor mismatch: {path}")


def verify_freeze(expected_freeze: str, expected_audit: str) -> dict[str, Any]:
    if sha256(FREEZE) != expected_freeze or FREEZE_SIDECAR.read_text(encoding="ascii").strip() != expected_freeze:
        raise FusionError("fusion freeze pin mismatch")
    if sha256(AUDIT) != expected_audit:
        raise FusionError("fusion audit pin mismatch")
    freeze, audit = read_json(FREEZE), read_json(AUDIT)
    if freeze.get("state") != "frozen_before_generic_atomic_completion" or audit.get("status") != "PASS" or audit.get("freeze_sha256") != expected_freeze:
        raise FusionError("freeze/audit semantic mismatch")
    for group_name in ("artifacts", "implementation"):
        group = freeze.get(group_name)
        if type(group) is not dict:
            raise FusionError("descriptor group missing")
        for value in group.values():
            _verify_descriptor(value)
    if (HERE / str(freeze["implementation"]["selective_fusion"]["path"])).resolve() != Path(__file__).resolve():
        raise FusionError("runtime implementation is not frozen implementation")
    return freeze


def _alignment_closure(
    public: list[dict[str, Any]], base: list[dict[str, Any]], decisions: list[dict[str, Any]],
    alignment: list[dict[str, Any]], generic_queue: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    base_ids = [row.get("task_id") for row in base]
    public_ids = [row.get("controller_id") for row in public]
    decision_ids = [row.get("runtime_alignment_id") for row in decisions]
    alignment_ids = [row.get("task_id") for row in alignment]
    prediction_ids = [row.get("task_id") for row in predictions]
    sequences: Iterable[tuple[str, list[Any], int]] = (
        ("base", base_ids, 274), ("public", public_ids, 274), ("decision", decision_ids, 274),
        ("alignment", alignment_ids, 256), ("prediction", prediction_ids, 256),
    )
    for name, values, expected in sequences:
        if len(values) != expected or len(set(values)) != expected or any(type(value) is not str or not value for value in values):
            raise FusionError(f"{name} identity closure mismatch")
    if not (base_ids == public_ids == decision_ids) or alignment_ids != prediction_ids:
        raise FusionError("outer order mismatch")
    generic_ids = [row["runtime_alignment_id"] for row in decisions if row.get("branch") == "generic_qwen35_9b"]
    if len(generic_ids) != 256 or generic_ids != alignment_ids or len(generic_queue) != 256:
        raise FusionError("generic partition/order mismatch")
    public_by_id = dict(zip(public_ids, public))
    base_by_id = dict(zip(base_ids, base))
    prediction_by_id = dict(zip(prediction_ids, predictions))
    answer_type_by_id: dict[str, str] = {}
    for task_id, public_row, content_row in zip(alignment_ids, (public_by_id[value] for value in alignment_ids), generic_queue):
        answer_type = public_row.get("answer_type")
        if answer_type not in ANSWER_TYPES or content_row.get("answer_type") != answer_type:
            raise FusionError("answer type alignment mismatch")
        answer_type_by_id[task_id] = str(answer_type)
    return base_ids, base_by_id, prediction_by_id, answer_type_by_id


def compose(expected_freeze: str, expected_audit: str, expected_completion: str, expected_predictions: str) -> dict[str, Any]:
    freeze = verify_freeze(expected_freeze, expected_audit)
    if OUTPUT.exists() or MANIFEST.exists():
        raise FusionError("fusion output already exists")
    if sha256(COMPLETION) != expected_completion or sha256(PREDICTIONS) != expected_predictions:
        raise FusionError("candidate completion/prediction pin mismatch")
    completion = read_json(COMPLETION)
    if completion.get("freeze_sha256") != CANDIDATE_FREEZE_SHA or completion.get("rows") != 256 or completion.get("predictions", {}).get("sha256") != expected_predictions or completion.get("gold_opened") is not False or completion.get("outcomes_opened") is not False:
        raise FusionError("candidate completion closure mismatch")
    public, base, decisions = read_jsonl(PUBLIC_QUEUE), read_jsonl(BASE251), read_jsonl(DECISIONS)
    alignment, generic_queue, predictions = read_jsonl(ALIGNMENT), read_jsonl(GENERIC_QUEUE), read_jsonl(PREDICTIONS)
    base_ids, base_by_id, prediction_by_id, answer_type_by_id = _alignment_closure(public, base, decisions, alignment, generic_queue, predictions)
    normalizer = _load_normalizer()
    lines: list[bytes] = []
    selected_generic = selected_baseline = generic_invalid = 0
    generic_id_set = set(answer_type_by_id)
    for task_id in base_ids:
        baseline_row = base_by_id[task_id]
        if task_id not in generic_id_set:
            selected = "certified_noid_baseline251"
            answer = str(baseline_row["final_answer"])
            reason = "preserve_certified_noid"
            selected_baseline += 1
        else:
            prediction = prediction_by_id[task_id]
            answer_type = answer_type_by_id[task_id]
            projection = generic_projection(prediction, task_id, answer_type)
            action = select_observable(answer_type=answer_type, baseline_answer=baseline_row.get("final_answer"), generic_projection=projection, normalizer=normalizer)
            if action["selected"] == "generic":
                selected = "generic_valid_over_unparseable_baseline"
                answer = official_generic_answer(prediction, answer_type)
                selected_generic += 1
            else:
                selected = "baseline251"
                answer = str(baseline_row["final_answer"])
                selected_baseline += 1
                generic_invalid += int(not action["generic_valid"])
            reason = str(action["reason"])
        lines.append(canonical({
            "task_id": task_id,
            "final_answer": answer,
            "condition": "identity_free_selective_fusion_v1",
            "generation": {
                "selected_branch": selected,
                "selection_reason": reason,
                "identity_used_by_branch_selector": False,
                "selector_projection_fields": ["answer_type", "baseline_answer", "generic_projection_without_identity"],
                "gold_access": False,
                "outcome_access": False,
                "fusion_rule_freeze_sha256": expected_freeze,
                "generic_candidate_freeze_sha256": CANDIDATE_FREEZE_SHA,
            },
        }))
    payload = b"".join(lines)
    exclusive_bytes(OUTPUT, payload)
    value = {
        "schema_version": "maxim274-identity-free-selective-fusion-completion-v1",
        "fusion_rule_freeze_sha256": expected_freeze,
        "candidate_completion_sha256": expected_completion,
        "candidate_predictions_sha256": expected_predictions,
        "base251_sha256": BASE251_SHA,
        "output_sha256": sha256(OUTPUT),
        "rows": 274,
        "selected_generic_rows": selected_generic,
        "selected_baseline_rows": selected_baseline,
        "invalid_generic_rows_fell_back": generic_invalid,
        "precomputed_maximum_generic_switch_candidates": freeze["precompletion_census"]["baseline_unparseable_generic_rows"],
        "identity_used_by_branch_selector": False,
        "semantic_correctness_claimed": False,
        "gold_opened": False,
        "outcomes_opened": False,
    }
    exclusive_bytes(MANIFEST, canonical(value))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    parser.add_argument("--expected-candidate-completion-sha256", required=True)
    parser.add_argument("--expected-candidate-predictions-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.expected_freeze_sha256, args.expected_independent_audit_sha256, args.expected_candidate_completion_sha256, args.expected_candidate_predictions_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

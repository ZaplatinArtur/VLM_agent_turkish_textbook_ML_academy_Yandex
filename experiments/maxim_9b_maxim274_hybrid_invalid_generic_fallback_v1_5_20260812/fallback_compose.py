"""Binary-exact successor to frozen fallback V1.4 for Windows-safe composition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
V14 = EXPERIMENTS / "maxim_9b_maxim274_hybrid_invalid_generic_fallback_v1_4_20260812"
CANDIDATE = (
    EXPERIMENTS
    / "maxim_9b_maxim274_hybrid_generic_siliconflow_nonstream_v1_3_20260812"
)
HYBRID = EXPERIMENTS / "maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812"
BASE240 = (
    EXPERIMENTS
    / "maxim_9b_baseline_selector_v1"
    / "compositor_output_v1_2"
    / "primary_solver.jsonl"
)
PREDICTIONS = CANDIDATE / "runs" / "generic_predictions_256.jsonl"
COMPLETION = CANDIDATE / "COMPLETION.json"
ALIGNMENT = CANDIDATE / "frozen" / "outer_alignment_256.jsonl"
QUEUE = CANDIDATE / "frozen" / "queue_content_only_256.jsonl"
DECISIONS = HYBRID / "runs" / "maxim274" / "route_decisions.jsonl"
OUTPUT = HERE / "runs" / "hybrid_solver_274_invalid_fallback.jsonl"
MANIFEST = HERE / "FALLBACK_COMPOSITION_COMPLETION.json"
FREEZE = HERE / "FALLBACK_RULE_FREEZE.json"
FREEZE_SHA_FILE = HERE / "FALLBACK_RULE_FREEZE_SHA256.txt"
AUDIT = HERE / "INDEPENDENT_AUDIT.json"

CANDIDATE_FREEZE_SHA = "7096b70b25281da114d70956fd8f7a468372eb315c955584fe83217b30bc630d"
CANDIDATE_AUDIT_SHA = "943c17d8f4a9be0876b38c170ac554db751df49c0c243146795a1e8024445570"
ALIGNMENT_SHA = "205b627f8eddf7376068787e12b7f5ffda7cf13f610d9240cbc1077e99b8f0b8"
QUEUE_SHA = "c8e16e97a32ebea4a1a43a4273ea0109d44ce1da5e067d32223872cd53dd0bb2"
DECISIONS_SHA = "0f6a31ec8d862ba67f61a8d90e96a0bd1e2a77d071a51ff1a36c512eeaea974c"
BASE240_SHA = "09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef"
V14_FREEZE_SHA = "3d9f816ecd0be04d120947a5c59f7b31364d229393a03650255b533e51606a2b"
V14_AUDIT_SHA = "7bc7f86aa392ca054cfe33062e4ac1d4fee6bf10a804b07f5b3e347dfb5c8e41"
V14_IMPLEMENTATION_SHA = "f22643a7b369a922e84a4fab73a93e734f9182474c054970146bb54ea2c056f3"

AUDIT_CHECKS = {
    "all_274_decision_ids_unique_string_exact_base_order",
    "base240_exact_raw_fallback",
    "binary_exact_writes_and_postwrite_hash",
    "candidate_outputs_absent_at_freeze",
    "choice_outputs_exact_option_label",
    "freeze_closure",
    "full_denominator_274",
    "generic_decision_alignment_set_exact",
    "hostile_choice_labels_rejected",
    "hostile_duplicate_missing_ids_abort",
    "hostile_windows_newline_regression",
    "identity_mismatch_aborts",
    "invalid_contract_complete",
    "no_gold_outcomes_score",
    "nonchoice_outputs_final_answer",
    "pre_outcome_rule",
    "runtime_self_implementation_sha",
    "tests_pass",
    "v1_4_pass_but_superseded_preuse_windows_defect_truthful",
    "valid_generic_semantic_risk_disclosed",
}


class FallbackError(RuntimeError):
    pass


def stable_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FallbackError(f"required file missing: {path}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != before.st_size
    ):
        raise FallbackError(f"file changed during read: {path}")
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(stable_bytes(path))


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(stable_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FallbackError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise FallbackError(f"JSON root is not object: {path}")
    return value


def jsonl_with_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(keepends=True), 1):
        if not raw.strip():
            continue
        if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
            raise FallbackError(f"JSONL row {number} is not exact terminal LF: {path}")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FallbackError(f"invalid JSONL row {number}: {path}") from exc
        if type(value) is not dict:
            raise FallbackError("JSONL row is not object")
        rows.append((value, raw))
    return rows


def exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if stable_bytes(path) != data or sha256(path) != sha256_bytes(data):
        raise FallbackError("binary output changed exact payload bytes")


def _load_v14() -> Any:
    source = V14 / "fallback_compose.py"
    if sha256(source) != V14_IMPLEMENTATION_SHA:
        raise FallbackError("V1.4 pure validation implementation SHA mismatch")
    spec = importlib.util.spec_from_file_location("maxim_fallback_v14_pure", source)
    if spec is None or spec.loader is None:
        raise FallbackError("cannot load V1.4 pure validation implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_freeze(expected_freeze: str, expected_audit: str) -> dict[str, Any]:
    if (
        sha256(FREEZE) != expected_freeze
        or FREEZE_SHA_FILE.read_text(encoding="ascii").strip() != expected_freeze
        or sha256(AUDIT) != expected_audit
    ):
        raise FallbackError("fallback V1.5 freeze/audit pin mismatch")
    freeze = read_json(FREEZE)
    audit = read_json(AUDIT)
    if (
        freeze.get("schema_version")
        != "maxim-invalid-generic-fallback-rule-freeze-v1.5"
        or freeze.get("state") != "frozen_before_candidate_completion"
        or audit.get("schema_version")
        != "maxim-invalid-generic-fallback-independent-audit-v1.5"
        or audit.get("status") != "PASS"
        or audit.get("freeze_sha256") != expected_freeze
        or set(audit.get("checks", {})) != AUDIT_CHECKS
        or any(value is not True for value in audit["checks"].values())
        or audit.get("guards")
        != {
            "api_called": False,
            "candidate_completion_opened": False,
            "candidate_predictions_opened": False,
            "gold_opened": False,
            "outcomes_opened": False,
            "score_opened": False,
        }
    ):
        raise FallbackError("fallback V1.5 freeze/audit semantic mismatch")
    for group_name in ("artifacts", "implementation"):
        group = freeze.get(group_name)
        if type(group) is not dict or not group:
            raise FallbackError("frozen descriptor group missing")
        for descriptor in group.values():
            if (
                type(descriptor) is not dict
                or set(descriptor) != {"path", "sha256", "size"}
            ):
                raise FallbackError("frozen descriptor schema mismatch")
            path = (HERE / descriptor["path"]).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != descriptor["size"]
                or sha256(path) != descriptor["sha256"]
            ):
                raise FallbackError("frozen descriptor closure mismatch")
    implementation = (HERE / freeze["implementation"]["fallback_compose"]["path"]).resolve()
    if implementation != Path(__file__).resolve():
        raise FallbackError("executing compositor is not frozen V1.5 implementation")
    return freeze


def compose(
    expected_freeze: str,
    expected_audit: str,
    expected_completion: str,
    expected_predictions: str,
) -> dict[str, Any]:
    verify_freeze(expected_freeze, expected_audit)
    if OUTPUT.exists() or MANIFEST.exists():
        raise FallbackError("fallback output exists")
    fixed = (
        (CANDIDATE / "EXECUTION_FREEZE.json", CANDIDATE_FREEZE_SHA),
        (CANDIDATE / "INDEPENDENT_AUDIT.json", CANDIDATE_AUDIT_SHA),
        (ALIGNMENT, ALIGNMENT_SHA),
        (QUEUE, QUEUE_SHA),
        (DECISIONS, DECISIONS_SHA),
        (BASE240, BASE240_SHA),
        (V14 / "FALLBACK_RULE_FREEZE.json", V14_FREEZE_SHA),
        (V14 / "INDEPENDENT_AUDIT.json", V14_AUDIT_SHA),
    )
    if any(sha256(path) != digest for path, digest in fixed):
        raise FallbackError("fallback ancestry pin mismatch")
    if sha256(COMPLETION) != expected_completion or sha256(PREDICTIONS) != expected_predictions:
        raise FallbackError("candidate completion/predictions external pin mismatch")
    completion = read_json(COMPLETION)
    if (
        completion.get("freeze_sha256") != CANDIDATE_FREEZE_SHA
        or completion.get("rows") != 256
        or completion.get("predictions", {}).get("sha256") != expected_predictions
        or completion.get("gold_opened") is not False
        or completion.get("outcomes_opened") is not False
    ):
        raise FallbackError("candidate completion closure mismatch")

    legacy = _load_v14()
    predictions = [row for row, _ in jsonl_with_raw(PREDICTIONS)]
    alignment = [row for row, _ in jsonl_with_raw(ALIGNMENT)]
    queue = [row for row, _ in jsonl_with_raw(QUEUE)]
    decisions = [row for row, _ in jsonl_with_raw(DECISIONS)]
    base_rows = jsonl_with_raw(BASE240)
    if (len(predictions), len(alignment), len(queue), len(decisions), len(base_rows)) != (
        256,
        256,
        256,
        274,
        274,
    ):
        raise FallbackError("denominator mismatch")
    base_order = [row.get("task_id") for row, _ in base_rows]
    if any(type(value) is not str or not value for value in base_order):
        raise FallbackError("base240 ID mismatch")
    base = {row.get("task_id"): (row, raw) for row, raw in base_rows}
    if len(base) != 274:
        raise FallbackError("base240 identity mismatch")
    outer_ids = [outer.get("task_id") for outer in alignment]
    generic_ids = legacy._validate_identity_closure(decisions, base_order, outer_ids)
    prediction_by_id: dict[str, dict[str, Any]] = {}
    answer_type_by_id: dict[str, str] = {}
    validity: dict[str, bool] = {}
    for prediction, outer, content in zip(predictions, alignment, queue):
        task_id = outer.get("task_id")
        if type(task_id) is not str or task_id in prediction_by_id:
            raise FallbackError("outer alignment mismatch")
        prediction_by_id[task_id] = prediction
        answer_type_by_id[task_id] = str(content.get("answer_type"))
        validity[task_id] = legacy._valid_prediction(
            prediction, task_id, answer_type_by_id[task_id]
        )
    if set(prediction_by_id) != generic_ids or len(generic_ids) != 256:
        raise FallbackError("generic decision/alignment ID set mismatch")

    lines: list[bytes] = []
    invalid_raw_by_id: dict[str, bytes] = {}
    certified = valid_generic = invalid_fallback = 0
    for decision in decisions:
        task_id, branch = decision.get("runtime_alignment_id"), decision.get("branch")
        if task_id not in base:
            raise FallbackError("decision/base membership mismatch")
        if branch == "certified_noid":
            answer = str(decision["action"]["answer"])
            lines.append(
                canonical(
                    {
                        "task_id": task_id,
                        "final_answer": answer,
                        "condition": "strict_noid_db_generic_hybrid_v3_invalid_fallback",
                        "generation": {
                            "selected_branch": "certified_noid",
                            "gold_access": False,
                            "identity_used_by_branch_selector": False,
                            "fallback_rule_freeze_sha256": expected_freeze,
                        },
                    }
                )
            )
            certified += 1
        elif branch == "generic_qwen35_9b" and validity.get(task_id) is True:
            lines.append(
                canonical(
                    {
                        "task_id": task_id,
                        "final_answer": legacy._official_answer(
                            prediction_by_id[task_id], answer_type_by_id[task_id]
                        ),
                        "condition": "strict_noid_db_generic_hybrid_v3_invalid_fallback",
                        "generation": {
                            "selected_branch": "qwen35_9b_frozen_candidate_valid",
                            "gold_access": False,
                            "identity_used_by_branch_selector": False,
                            "fallback_rule_freeze_sha256": expected_freeze,
                            "generic_candidate_freeze_sha256": CANDIDATE_FREEZE_SHA,
                        },
                    }
                )
            )
            valid_generic += 1
        elif branch == "generic_qwen35_9b":
            raw = base[task_id][1]
            lines.append(raw)
            invalid_raw_by_id[task_id] = raw
            invalid_fallback += 1
        else:
            raise FallbackError("unexpected frozen branch")
    payload = b"".join(lines)
    if b"\r" in payload or payload.count(b"\n") != 274:
        raise FallbackError("composed payload is not exact 274-row LF JSONL")
    exclusive_bytes(OUTPUT, payload)
    written = jsonl_with_raw(OUTPUT)
    written_raw = {row["task_id"]: raw for row, raw in written}
    if any(written_raw.get(task_id) != raw for task_id, raw in invalid_raw_by_id.items()):
        raise FallbackError("postwrite base240 raw fallback byte mismatch")
    value = {
        "schema_version": "maxim-invalid-generic-fallback-completion-v1.5",
        "fallback_rule_freeze_sha256": expected_freeze,
        "candidate_completion_sha256": expected_completion,
        "candidate_predictions_sha256": expected_predictions,
        "base240_sha256": BASE240_SHA,
        "output_sha256": sha256(OUTPUT),
        "rows": 274,
        "certified_noid_rows": certified,
        "valid_generic_rows": valid_generic,
        "invalid_generic_base240_exact_raw_rows": invalid_fallback,
        "choice_projection": "option_label",
        "nonchoice_projection": "final_answer",
        "identity_closure": "274 unique decision IDs exact base240 order; 256 generic IDs exact alignment set",
        "binary_exact_write": True,
        "terminal_newline": "LF",
        "carriage_returns": 0,
        "invalid_raw_rows_postwrite_byte_verified": True,
        "supersedes_v1_4_freeze_sha256": V14_FREEZE_SHA,
        "gold_opened": False,
        "outcomes_opened": False,
    }
    manifest_data = canonical(value)
    exclusive_bytes(MANIFEST, manifest_data)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    parser.add_argument("--expected-candidate-completion-sha256", required=True)
    parser.add_argument("--expected-candidate-predictions-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compose(
                args.expected_freeze_sha256,
                args.expected_independent_audit_sha256,
                args.expected_candidate_completion_sha256,
                args.expected_candidate_predictions_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

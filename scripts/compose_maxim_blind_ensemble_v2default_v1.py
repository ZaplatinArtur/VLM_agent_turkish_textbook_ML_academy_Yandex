"""Compose preregistered gold-blind conservative selectors over frozen solvers.

The default row is always copied byte-for-byte in JSON value space from Meta-V2.1.
An override is permitted only by the selected profile and only after all generic
route/confidence/evidence/format gates are revalidated.  Task IDs are used solely
to align immutable rows; they never participate in a selection decision.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_ROWS = 274
PROFILE_SCHEMA = "maxim-blind-ensemble-v2-default-profile-v1"
AUDIT_SCHEMA = "maxim-blind-ensemble-v2-default-audit-v1"
MANIFEST_SCHEMA = "maxim-blind-ensemble-v2-default-composition-v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
V2_ROUTER_FAILCLOSED_REASONS = {
    "explicit_abstention_router_fallback",
    "format_not_verified_router_fallback",
    "confidence_gate_router_fallback",
    "evidence_gate_router_fallback",
    "verifier_error_router_fallback",
}


class CompositionError(RuntimeError):
    """Raised when an immutable source or preregistered contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompositionError(f"{path}: expected one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompositionError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def canonical_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _float_at_least(value: Any, minimum: float) -> bool:
    try:
        return float(value) >= minimum
    except (TypeError, ValueError):
        return False


def _nonempty_items(value: Any, minimum: int) -> bool:
    return isinstance(value, list) and sum(
        bool(str(item or "").strip()) for item in value
    ) >= minimum


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _index(
    rows: Sequence[Mapping[str, Any]], label: str, *, expected_rows: int = EXPECTED_ROWS
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    if len(rows) != expected_rows:
        raise CompositionError(
            f"{label}: expected {expected_rows} rows, received {len(rows)}"
        )
    index: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for row_number, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise CompositionError(f"{label}:{row_number}: missing task_id")
        if task_id in index:
            raise CompositionError(f"{label}: duplicate task_id")
        index[task_id] = row
        order.append(task_id)
    return index, order


def _assert_source_hashes(
    profile: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    bindings = _mapping(profile.get("source_bindings"))
    if bindings is None:
        raise CompositionError("profile source_bindings missing")
    if set(bindings) != set(paths):
        raise CompositionError("profile source binding names do not match CLI inputs")
    for name, path in paths.items():
        binding = _mapping(bindings.get(name))
        if binding is None:
            raise CompositionError(f"profile source binding missing: {name}")
        expected = str(binding.get("sha256") or "")
        if not HEX64_RE.fullmatch(expected):
            raise CompositionError(f"profile source SHA invalid: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise CompositionError(
                f"{name} SHA mismatch: expected={expected}, actual={actual}"
            )


def _validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise CompositionError("profile schema mismatch")
    if profile.get("frozen_before_source_row_values_read") is not True:
        raise CompositionError("profile is not declared frozen before row-value read")
    if profile.get("gold_access") is not False:
        raise CompositionError("profile gold_access must be false")
    if profile.get("score_or_judge_inputs_allowed") is not False:
        raise CompositionError("profile must forbid score and judge inputs")
    if profile.get("selection_uses_task_id") is not False:
        raise CompositionError("profile must forbid task-ID selection")
    if profile.get("default_source") != "meta_v21":
        raise CompositionError("default source must be Meta-V2.1")
    if profile.get("policy_id") not in {
        "triple_agreement_v3_active",
        "v3_repairs_v2_failclosed",
        "active_repairs_v2_failclosed",
    }:
        raise CompositionError("unknown policy_id")
    thresholds = _mapping(profile.get("thresholds"))
    if thresholds is None:
        raise CompositionError("thresholds missing")
    exact = {
        "v3_min_confidence": 0.90,
        "v3_min_decisive_evidence": 2,
        "active_min_verifier_confidence": 0.90,
        "active_min_locator_confidence": 0.80,
        "active_min_region_confidence": 0.70,
        "active_min_visible_facts": 2,
        "active_min_verification_checks": 2,
    }
    if dict(thresholds) != exact:
        raise CompositionError("profile thresholds differ from frozen constants")


def _audit_row_bound(
    audit: Mapping[str, Any], source_row: Mapping[str, Any]
) -> bool:
    return (
        str(audit.get("task_id") or "") == str(source_row.get("task_id") or "")
        and str(audit.get("output_row_sha256") or "") == stable_sha256(source_row)
        and audit.get("gold_access") is False
        and audit.get("task_id_or_subject_used_for_selection") is False
    )


def v2_is_generic_failclosed(
    row: Mapping[str, Any],
    audit: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not _audit_row_bound(audit, row):
        failures.append("v2_audit_binding_invalid")
    if str(verifier.get("task_id") or "") != str(row.get("task_id") or ""):
        failures.append("v2_verifier_binding_invalid")
    selection = _mapping(verifier.get("selection"))
    if selection is None:
        failures.append("v2_selection_missing")
        return False, failures
    if selection.get("gold_access") is not False:
        failures.append("v2_selection_not_gold_blind")
    if selection.get("selected_source") != "router":
        failures.append("v2_not_router_failclosed")
    reason = str(selection.get("reason") or "")
    if reason not in V2_ROUTER_FAILCLOSED_REASONS:
        failures.append("v2_router_reason_not_generic_failclosed")
    decision = str(audit.get("decision") or "")
    if decision == "numeric_choice_token_compat_recovered":
        failures.append("v21_compat_recovered_not_failclosed")
    elif decision not in {
        "unchanged_v2_content_exact",
        "nonrecoverable_error_exact_router",
    }:
        failures.append("v21_audit_decision_unrecognized")
    if canonical_answer(selection.get("applied_final_answer")) != canonical_answer(
        row.get("final_answer")
    ):
        failures.append("v2_applied_answer_mismatch")
    if not canonical_answer(row.get("final_answer")):
        failures.append("v2_default_answer_empty")
    return not failures, failures


def v3_is_high_confidence_supported(
    row: Mapping[str, Any],
    audit: Mapping[str, Any],
    verifier: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not _audit_row_bound(audit, row):
        failures.append("v31_audit_binding_invalid")
    if str(verifier.get("task_id") or "") != str(row.get("task_id") or ""):
        failures.append("v3_verifier_binding_invalid")
    if row.get("error") not in (None, ""):
        failures.append("v31_row_error")
    if not canonical_answer(row.get("final_answer")):
        failures.append("v31_answer_empty")
    min_confidence = float(thresholds["v3_min_confidence"])
    min_evidence = int(thresholds["v3_min_decisive_evidence"])
    decision = str(audit.get("decision") or "")
    if decision == "unchanged_v3_content_exact":
        if verifier.get("error") not in (None, ""):
            failures.append("v3_verifier_error")
        selection = _mapping(verifier.get("selection"))
        verdict = _mapping(verifier.get("verdict"))
        call = _mapping(verifier.get("call"))
        if selection is None:
            failures.append("v3_selection_missing")
        else:
            if selection.get("gold_access") is not False:
                failures.append("v3_selection_not_gold_blind")
            if selection.get("selected_source") != "meta_verifier":
                failures.append("v3_not_meta_selected")
            if selection.get("reason") != "valid_supported_meta_answer":
                failures.append("v3_selection_reason_invalid")
            if canonical_answer(selection.get("applied_final_answer")) != canonical_answer(
                row.get("final_answer")
            ):
                failures.append("v3_applied_answer_mismatch")
        if verdict is None:
            failures.append("v3_verdict_missing")
        else:
            if verdict.get("abstain") is not False:
                failures.append("v3_abstained")
            if verdict.get("answer_format_verified") is not True:
                failures.append("v3_format_not_verified")
            if not _float_at_least(verdict.get("confidence"), min_confidence):
                failures.append("v3_confidence_below_0_90")
            if not _nonempty_items(verdict.get("decisive_evidence"), min_evidence):
                failures.append("v3_evidence_below_minimum")
            if canonical_answer(verdict.get("final_answer")) != canonical_answer(
                row.get("final_answer")
            ):
                failures.append("v3_verdict_answer_mismatch")
        if call is None:
            failures.append("v3_call_metadata_missing")
        else:
            if call.get("parse_error") not in (None, ""):
                failures.append("v3_call_parse_error")
            if call.get("recovered_partial") not in (False, None):
                failures.append("v3_call_recovered_partial")
            if call.get("finish_reason") != "stop":
                failures.append("v3_call_not_stop")
    elif decision == "numeric_choice_token_compat_recovered":
        verdict = _mapping(audit.get("raw_recovered_verdict"))
        generation = _mapping(row.get("generation"))
        if verdict is None:
            failures.append("v31_recovered_verdict_missing")
        else:
            if verdict.get("abstain") is not False:
                failures.append("v31_recovered_abstained")
            if verdict.get("answer_format_verified") is not True:
                failures.append("v31_recovered_format_not_verified")
            if not _float_at_least(verdict.get("confidence"), min_confidence):
                failures.append("v31_recovered_confidence_below_0_90")
            if not _nonempty_items(verdict.get("decisive_evidence"), min_evidence):
                failures.append("v31_recovered_evidence_below_minimum")
            if canonical_answer(verdict.get("final_answer")) != canonical_answer(
                row.get("final_answer")
            ):
                failures.append("v31_recovered_answer_mismatch")
        if generation is None:
            failures.append("v31_generation_missing")
        else:
            if generation.get("gold_access") is not False:
                failures.append("v31_generation_not_gold_blind")
            if generation.get("answer_format_verified") is not True:
                failures.append("v31_generation_format_not_verified")
            if not _float_at_least(generation.get("confidence"), min_confidence):
                failures.append("v31_generation_confidence_below_0_90")
            if generation.get("selection_reason") != (
                "valid_supported_meta_answer_choice_token_compat_v31"
            ):
                failures.append("v31_generation_selection_reason_invalid")
    else:
        failures.append("v31_not_supported_meta_route")
    return not failures, failures


def active_is_high_confidence_selected(
    row: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if row.get("error") not in (None, ""):
        failures.append("active_row_error")
    if not canonical_answer(row.get("final_answer")):
        failures.append("active_answer_empty")
    generation = _mapping(row.get("generation"))
    if generation is None:
        return False, ["active_generation_missing"]
    if generation.get("gold_access") is not False:
        failures.append("active_generation_not_gold_blind")
    composition = _mapping(generation.get("active_crop_failclosed_composition"))
    if composition is None:
        failures.append("active_composition_metadata_missing")
    else:
        if composition.get("selected_source") != "active_crop":
            failures.append("active_source_not_selected")
        if composition.get("gate_passed") is not True:
            failures.append("active_source_gate_not_passed")
        if composition.get("failed_clauses") != []:
            failures.append("active_source_has_failed_clauses")
        if not HEX64_RE.fullmatch(str(composition.get("candidate_row_sha256") or "")):
            failures.append("active_candidate_sha_missing")
        if not HEX64_RE.fullmatch(str(composition.get("fallback_row_sha256") or "")):
            failures.append("active_fallback_sha_missing")
    evidence = _mapping(generation.get("selection_evidence"))
    if evidence is None:
        failures.append("active_selection_evidence_missing")
    else:
        if evidence.get("baseline_supported") is not False:
            failures.append("active_baseline_not_refuted")
        if not _float_at_least(
            evidence.get("confidence"),
            float(thresholds["active_min_verifier_confidence"]),
        ):
            failures.append("active_verifier_confidence_below_0_90")
        for key in (
            "all_required_evidence_visible",
            "original_crop_consistent",
            "answer_format_verified",
        ):
            if evidence.get(key) is not True:
                failures.append(f"active_{key}_not_true")
        if not _nonempty_items(
            evidence.get("visible_facts"),
            int(thresholds["active_min_visible_facts"]),
        ):
            failures.append("active_visible_facts_below_minimum")
        if not _nonempty_items(
            evidence.get("verification_checks"),
            int(thresholds["active_min_verification_checks"]),
        ):
            failures.append("active_verification_checks_below_minimum")
    locator = _mapping(generation.get("locator"))
    if locator is None:
        failures.append("active_locator_missing")
    else:
        if not _float_at_least(
            locator.get("overall_confidence"),
            float(thresholds["active_min_locator_confidence"]),
        ):
            failures.append("active_locator_confidence_below_0_80")
        regions = locator.get("used_regions")
        if not isinstance(regions, list) or not 1 <= len(regions) <= 2:
            failures.append("active_region_count_invalid")
        else:
            for region in regions:
                if not isinstance(region, Mapping) or not _float_at_least(
                    region.get("confidence"),
                    float(thresholds["active_min_region_confidence"]),
                ):
                    failures.append("active_region_confidence_below_0_70")
                    break
    return not failures, failures


def compose(
    *,
    profile: Mapping[str, Any],
    v2_rows: Sequence[Mapping[str, Any]],
    v2_audits: Sequence[Mapping[str, Any]],
    v2_verifiers: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
    v3_audits: Sequence[Mapping[str, Any]],
    v3_verifiers: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    _validate_profile(profile)
    v2, order = _index(v2_rows, "Meta-V2.1")
    v2_audit, v2_audit_order = _index(v2_audits, "Meta-V2.1 audit")
    v2_verifier, v2_verifier_order = _index(v2_verifiers, "Meta-V2 verifier")
    v3, v3_order = _index(v3_rows, "Meta-V3.1")
    v3_audit, v3_audit_order = _index(v3_audits, "Meta-V3.1 audit")
    v3_verifier, v3_verifier_order = _index(v3_verifiers, "Meta-V3 verifier")
    active, active_order = _index(active_rows, "Active-Crop V2")
    for label, candidate_order in (
        ("Meta-V2.1 audit", v2_audit_order),
        ("Meta-V2 verifier", v2_verifier_order),
        ("Meta-V3.1", v3_order),
        ("Meta-V3.1 audit", v3_audit_order),
        ("Meta-V3 verifier", v3_verifier_order),
        ("Active-Crop V2", active_order),
    ):
        if candidate_order != order:
            raise CompositionError(f"{label}: task order differs from Meta-V2.1")

    thresholds = _mapping(profile.get("thresholds"))
    assert thresholds is not None
    policy = str(profile["policy_id"])
    outputs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    for queue_index, task_id in enumerate(order):
        default = v2[task_id]
        v3_candidate = v3[task_id]
        active_candidate = active[task_id]
        v2_failclosed, v2_failures = v2_is_generic_failclosed(
            default, v2_audit[task_id], v2_verifier[task_id]
        )
        v3_trusted, v3_failures = v3_is_high_confidence_supported(
            v3_candidate,
            v3_audit[task_id],
            v3_verifier[task_id],
            thresholds,
        )
        active_trusted, active_failures = active_is_high_confidence_selected(
            active_candidate, thresholds
        )
        v2_answer = canonical_answer(default.get("final_answer"))
        v3_answer = canonical_answer(v3_candidate.get("final_answer"))
        active_answer = canonical_answer(active_candidate.get("final_answer"))
        selected_source = "meta_v21"
        decision = "default_exact_meta_v21"
        chosen = default
        if policy == "triple_agreement_v3_active":
            if (
                v3_trusted
                and active_trusted
                and v3_answer
                and v3_answer == active_answer
                and v3_answer != v2_answer
            ):
                selected_source = "meta_v31"
                decision = "override_v3_active_agree_against_v21"
                chosen = v3_candidate
        elif policy == "v3_repairs_v2_failclosed":
            if v2_failclosed and v3_trusted and v3_answer and v3_answer != v2_answer:
                selected_source = "meta_v31"
                decision = "override_v31_repairs_v21_failclosed"
                chosen = v3_candidate
        elif policy == "active_repairs_v2_failclosed":
            if (
                v2_failclosed
                and active_trusted
                and active_answer
                and active_answer != v2_answer
            ):
                selected_source = "active_crop_v2"
                decision = "override_active_repairs_v21_failclosed"
                chosen = active_candidate
        else:  # Defensive; _validate_profile already rejects this.
            raise CompositionError("unreachable unknown policy")

        output = copy.deepcopy(dict(chosen))
        if output != chosen:
            raise CompositionError("content-exact source copy failed")
        counts[decision] += 1
        outputs.append(output)
        audits.append(
            {
                "schema_version": AUDIT_SCHEMA,
                "queue_index": queue_index,
                "task_id": task_id,
                "policy_id": policy,
                "selected_source": selected_source,
                "decision": decision,
                "default_row_sha256": stable_sha256(default),
                "v31_row_sha256": stable_sha256(v3_candidate),
                "active_crop_v2_row_sha256": stable_sha256(active_candidate),
                "output_row_sha256": stable_sha256(output),
                "generic_gate_state": {
                    "v21_generic_failclosed": v2_failclosed,
                    "v31_high_confidence_supported": v3_trusted,
                    "active_crop_high_confidence_selected": active_trusted,
                    "v31_active_agree": bool(v3_answer and v3_answer == active_answer),
                    "v31_differs_from_v21": bool(v3_answer and v3_answer != v2_answer),
                    "active_differs_from_v21": bool(
                        active_answer and active_answer != v2_answer
                    ),
                },
                "failed_generic_gates": {
                    "v21": v2_failures,
                    "v31": v3_failures,
                    "active_crop_v2": active_failures,
                },
                "gold_access": False,
                "score_or_judge_access": False,
                "task_id_used_for_alignment_only": True,
                "task_id_or_subject_used_for_selection": False,
            }
        )
    return outputs, audits, dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--v2-solver", type=Path, required=True)
    parser.add_argument("--v2-audit", type=Path, required=True)
    parser.add_argument("--v2-verifier", type=Path, required=True)
    parser.add_argument("--v3-solver", type=Path, required=True)
    parser.add_argument("--v3-audit", type=Path, required=True)
    parser.add_argument("--v3-verifier", type=Path, required=True)
    parser.add_argument("--active-solver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    profile = load_json(args.profile)
    _validate_profile(profile)
    source_paths = {
        "meta_v21_solver": args.v2_solver,
        "meta_v21_audit": args.v2_audit,
        "meta_v2_verifier": args.v2_verifier,
        "meta_v31_solver": args.v3_solver,
        "meta_v31_audit": args.v3_audit,
        "meta_v3_verifier": args.v3_verifier,
        "active_crop_v2_solver": args.active_solver,
    }
    _assert_source_hashes(profile, source_paths)
    outputs, audits, decision_counts = compose(
        profile=profile,
        v2_rows=load_jsonl(args.v2_solver),
        v2_audits=load_jsonl(args.v2_audit),
        v2_verifiers=load_jsonl(args.v2_verifier),
        v3_rows=load_jsonl(args.v3_solver),
        v3_audits=load_jsonl(args.v3_audit),
        v3_verifiers=load_jsonl(args.v3_verifier),
        active_rows=load_jsonl(args.active_solver),
    )
    solver_path = args.output_dir / "solver.jsonl"
    audit_path = args.output_dir / "selection_audit.jsonl"
    write_jsonl(solver_path, outputs)
    write_jsonl(audit_path, audits)
    overrides = sum(
        count
        for decision, count in decision_counts.items()
        if decision != "default_exact_meta_v21"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "condition": profile.get("condition"),
        "policy_id": profile.get("policy_id"),
        "rows": len(outputs),
        "complete": len(outputs) == EXPECTED_ROWS,
        "gold_access": False,
        "score_or_judge_access": False,
        "selection_uses_task_id": False,
        "default_source": "meta_v21",
        "default_copy_mode": "exact_json_value_copy",
        "override_copy_mode": "exact_json_value_copy",
        "override_rows": overrides,
        "default_rows": len(outputs) - overrides,
        "decision_counts": decision_counts,
        "profile": {
            "path": str(args.profile),
            "sha256": sha256_file(args.profile),
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "artifacts": {
            "solver": {"path": str(solver_path), "sha256": sha256_file(solver_path)},
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        },
    }
    manifest_path = args.output_dir / "composition_manifest.json"
    write_json(manifest_path, manifest)
    sums = [
        f"{sha256_file(solver_path)}  solver.jsonl",
        f"{sha256_file(audit_path)}  selection_audit.jsonl",
        f"{sha256_file(manifest_path)}  composition_manifest.json",
    ]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "condition": profile.get("condition"),
                "rows": len(outputs),
                "override_rows": overrides,
                "decision_counts": decision_counts,
                "solver_sha256": sha256_file(solver_path),
                "audit_sha256": sha256_file(audit_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

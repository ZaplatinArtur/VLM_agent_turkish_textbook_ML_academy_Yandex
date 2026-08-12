"""No-key scorer using base251 image verdicts after exact-byte reuse proof.

This module does not score on import. Private benchmark/judge parsing requires
the explicit ``--execute-private-score`` flag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import compose
import protocol


def require_exact_mechanical_payload(actual: bytes) -> dict[str, int | bool]:
    sources = compose.validate_sources()
    expected, census = compose.compose_payload(sources["base"], sources["selected"], sources["image_ids"])
    if actual != expected:
        raise protocol.ProtocolError("solver is not the mechanically recomputed frozen payload")
    return census


def verify_reuse(expected_freeze: str, expected_audit: str, expected_completion: str, expected_solver: str) -> dict[str, object]:
    protocol.verify_own_protocol(expected_freeze, expected_audit)
    if protocol.sha256_file(protocol.COMPLETION) != expected_completion or protocol.sha256_file(protocol.OUTPUT) != expected_solver:
        raise protocol.ProtocolError("successor completion/solver pin mismatch")
    completion = protocol.read_json(protocol.COMPLETION)
    expected_values = {
        "schema_version": "maxim274-selective-fusion-byte-preserve-completion-v1.3",
        "freeze_sha256": expected_freeze,
        "independent_audit_sha256": expected_audit,
        "v1_1_completion_sha256": protocol.PINS["v1_1_completion"][1],
        "v1_1_solver_sha256": protocol.PINS["v1_1_solver"][1],
        "base251_solver_sha256": protocol.PINS["base251_solver"][1],
        "base251_image97_judge_sha256": protocol.PINS["base251_image97_judge"][1],
        "candidate_completion_sha256": protocol.PINS["candidate_completion"][1],
        "candidate_predictions_sha256": protocol.PINS["candidate_predictions"][1],
        "output_sha256": expected_solver,
        "rows": 274,
        "baseline_rows_copied_byte_exact": 272,
        "generic_rows_copied_from_v1_1_byte_exact": 2,
        "image97_rows_base251_byte_and_object_exact": 97,
        "image97_candidate_text_utf8_exact": True,
        "identity_used_for_branch_selection": False,
        "identity_used_postdecision_for_alignment": True,
        "gold_opened_by_compositor": False,
        "outcomes_opened_by_compositor": False,
        "semantic_tunables": 0,
    }
    if set(completion) != protocol.COMPLETION_KEYS or completion != expected_values:
        raise protocol.ProtocolError("successor completion does not bind image97 reuse")
    expected_census = require_exact_mechanical_payload(protocol.stable_bytes(protocol.OUTPUT))
    if expected_census != {
        "baseline_rows_copied_byte_exact": 272,
        "generic_rows_copied_from_v1_1_byte_exact": 2,
        "image97_rows_base251_byte_and_object_exact": 97,
        "image97_candidate_text_utf8_exact": True,
    }:
        raise protocol.ProtocolError("solver is not the mechanically recomputed frozen payload")
    base = protocol.jsonl_raw(protocol.BASE251_SOLVER)
    output = protocol.jsonl_raw(protocol.OUTPUT)
    image_ids = set(protocol.ordered_ids(protocol.jsonl_raw(protocol.IMAGE97_ALIGNMENT), "image97 alignment"))
    if protocol.ordered_ids(base, "base251") != protocol.ordered_ids(output, "successor"):
        raise protocol.ProtocolError("solver order mismatch")
    adapter = protocol.load_adapter()
    checked = 0
    for (base_row, base_raw), (out_row, out_raw) in zip(base, output):
        if base_row["task_id"] in image_ids:
            if base_raw != out_raw or base_row != out_row or adapter.candidate_text(base_row).encode("utf-8") != adapter.candidate_text(out_row).encode("utf-8"):
                raise protocol.ProtocolError("image97 verdict reuse is unsafe")
            checked += 1
    if checked != 97:
        raise protocol.ProtocolError("image97 reuse denominator mismatch")
    return {"image97_rows_verified": checked, "image_judge_sha256": protocol.PINS["base251_image97_judge"][1]}


def execute_private(expected_freeze: str, expected_audit: str, expected_completion: str, expected_solver: str) -> dict[str, object]:
    if protocol.PRIVATE_RESULT.exists():
        raise protocol.ProtocolError("private result already exists")
    reuse = verify_reuse(expected_freeze, expected_audit, expected_completion, expected_solver)
    spec = importlib.util.spec_from_file_location("byte_preserve_official_scorer", protocol.SCORER)
    if spec is None or spec.loader is None:
        raise protocol.ProtocolError("cannot load official scorer")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    report = scorer.build_report(
        benchmark_path=protocol.BENCHMARK,
        solver_results_path=protocol.OUTPUT,
        image_judge_path=protocol.BASE251_IMAGE_JUDGE,
        baseline_judge_path=protocol.SCORER_BASELINE_JUDGE,
        expected_rows=274,
        expected_deterministic=177,
        expected_image_judge=97,
        expected_benchmark_sha256=protocol.PINS["benchmark"][1],
        expected_baseline_judge_sha256=protocol.PINS["scorer_baseline_judge"][1],
        label="identity_free_selective_fusion_byte_preserve_v1_3",
    )
    expected_provenance = {
        "benchmark": protocol.PINS["benchmark"][1],
        "solver_results": expected_solver,
        "image_judge": protocol.PINS["base251_image97_judge"][1],
        "frozen_page_rag_judge": protocol.PINS["scorer_baseline_judge"][1],
        "scorer": protocol.PINS["official_scorer"][1],
    }
    provenance = report.get("provenance")
    guardrails = report.get("guardrails")
    overall = report.get("overall")
    if (
        type(provenance) is not dict
        or {name: value.get("sha256") if type(value) is dict else None for name, value in provenance.items()} != expected_provenance
        or guardrails != {
            "benchmark_rows_verified": 274,
            "solver_rows_verified": 274,
            "baseline_rows_verified": 274,
            "image_judge_rows_supplied": 97,
            "image_judge_input_shape": "image_only",
            "task_id_sets_match": True,
            "duplicate_task_ids": 0,
            "forbidden_gold_fields_in_solver": 0,
            "explicit_nonfalse_generation_gold_access": 0,
            "frozen_sha_pins_checked": True,
        }
        or type(overall) is not dict
        or type(overall.get("new_correct")) is not int
        or not 0 <= overall["new_correct"] <= 274
        or overall.get("n") != 274
        or overall.get("new_accuracy") != round(overall["new_correct"] / 274, 6)
    ):
        raise protocol.ProtocolError("official scorer provenance/guardrail closure failed")
    correct = overall["new_correct"]
    result = {
        "schema_version": "maxim274-byte-preserve-private-result-v1.3",
        "freeze_sha256": expected_freeze,
        "independent_audit_sha256": expected_audit,
        "solver_sha256": expected_solver,
        "completion_sha256": expected_completion,
        "official_scorer_sha256": protocol.PINS["official_scorer"][1],
        **reuse,
        "correct": correct,
        "total": 274,
        "accuracy": correct / 274,
        "success_threshold_correct_at_least": protocol.SUCCESS_THRESHOLD_CORRECT,
        "threshold_met": protocol.threshold_met(correct),
        "per_row_outcomes_persisted": False,
        "api_calls": 0,
    }
    protocol.exclusive_bytes(protocol.PRIVATE_RESULT, protocol.canonical_json(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-solver-sha256", required=True)
    parser.add_argument("--execute-private-score", action="store_true")
    args = parser.parse_args()
    if not args.execute_private_score:
        raise SystemExit("private score not executed: pass --execute-private-score explicitly")
    print(json.dumps(execute_private(args.expected_freeze_sha256, args.expected_independent_audit_sha256, args.expected_completion_sha256, args.expected_solver_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

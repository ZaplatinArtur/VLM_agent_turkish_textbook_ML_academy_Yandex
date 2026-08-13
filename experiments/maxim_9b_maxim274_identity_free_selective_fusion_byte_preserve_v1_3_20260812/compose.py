"""Compose using only immutable V1.1 decisions and exact source row bytes."""

from __future__ import annotations

import argparse
import json

import protocol


def validate_sources() -> dict[str, object]:
    protocol.verify_pins()
    base = protocol.jsonl_raw(protocol.BASE251_SOLVER)
    selected = protocol.jsonl_raw(protocol.V11_SOLVER)
    image = protocol.jsonl_raw(protocol.IMAGE97_ALIGNMENT)
    base_ids = protocol.ordered_ids(base, "base251")
    selected_ids = protocol.ordered_ids(selected, "V1.1 result")
    image_ids = protocol.ordered_ids(image, "image97 alignment")
    if len(base) != 274 or base_ids != selected_ids or len(image) != 97 or not set(image_ids).issubset(base_ids):
        raise protocol.ProtocolError("denominator/order closure failed")
    completion = protocol.read_json(protocol.V11_COMPLETION)
    candidate = protocol.read_json(protocol.CANDIDATE_COMPLETION)
    if (
        completion.get("selected_baseline_rows") != 272
        or completion.get("selected_generic_rows") != 2
        or completion.get("identity_used_by_branch_selector") is not False
        or candidate.get("rows") != 256
        or candidate.get("predictions", {}).get("sha256") != protocol.PINS["candidate_predictions"][1]
        or candidate.get("gold_opened") is not False
        or candidate.get("outcomes_opened") is not False
    ):
        raise protocol.ProtocolError("atomic completion semantic closure failed")
    baseline_count = generic_count = 0
    generic_ids: set[str] = set()
    forbidden = {"reference_answer", "gold_answer", "expected_answer", "is_correct", "score", "new_correct", "baseline_correct", "verdict"}
    for (base_row, _), (selected_row, _) in zip(base, selected):
        if forbidden.intersection(base_row) or forbidden.intersection(selected_row):
            raise protocol.ProtocolError("solver input contains forbidden outcome/gold field")
        decision = protocol.branch(selected_row)
        if decision in protocol.BASELINE_BRANCHES:
            baseline_count += 1
            if selected_row.get("final_answer") != base_row.get("final_answer"):
                raise protocol.ProtocolError("V1.1 baseline decision changed final answer")
        else:
            generic_count += 1
            generic_ids.add(str(base_row["task_id"]))
    if (baseline_count, generic_count) != (272, 2) or generic_ids.intersection(image_ids):
        raise protocol.ProtocolError("branch census or image97 exclusion failed")
    return {"base": base, "selected": selected, "image_ids": image_ids}


def compose_payload(base, selected, image_ids: list[str]) -> tuple[bytes, dict[str, int | bool]]:
    image_set = set(image_ids)
    payload: list[bytes] = []
    baseline_count = generic_count = image_exact = 0
    adapter = protocol.load_adapter()
    for (base_row, base_raw), (selected_row, selected_raw) in zip(base, selected):
        if base_row["task_id"] != selected_row["task_id"]:
            raise protocol.ProtocolError("postdecision alignment mismatch")
        decision = protocol.branch(selected_row)
        if decision in protocol.BASELINE_BRANCHES:
            output_row, output_raw = base_row, base_raw
            baseline_count += 1
        else:
            output_row, output_raw = selected_row, selected_raw
            generic_count += 1
        if base_row["task_id"] in image_set:
            if output_raw != base_raw or output_row != base_row:
                raise protocol.ProtocolError("image97 row is not exact base251 bytes/object")
            if adapter.candidate_text(output_row).encode("utf-8") != adapter.candidate_text(base_row).encode("utf-8"):
                raise protocol.ProtocolError("image97 candidate_text bytes differ")
            image_exact += 1
        payload.append(output_raw)
    if (baseline_count, generic_count, image_exact) != (272, 2, 97):
        raise protocol.ProtocolError("output census mismatch")
    return b"".join(payload), {
        "baseline_rows_copied_byte_exact": baseline_count,
        "generic_rows_copied_from_v1_1_byte_exact": generic_count,
        "image97_rows_base251_byte_and_object_exact": image_exact,
        "image97_candidate_text_utf8_exact": True,
    }


def run(expected_freeze: str, expected_audit: str) -> dict[str, object]:
    if protocol.OUTPUT.exists() or protocol.COMPLETION.exists():
        raise protocol.ProtocolError("successor output already exists")
    freeze = protocol.verify_own_protocol(expected_freeze, expected_audit)
    if freeze.get("census") != {
        "rows": 274,
        "baseline_selected": 272,
        "generic_selected": 2,
        "image97_rows": 97,
        "generic_selected_inside_image97": 0,
        "expected_image97_candidate_text_byte_matches": 97,
    }:
        raise protocol.ProtocolError("frozen census mismatch")
    sources = validate_sources()
    payload, census = compose_payload(sources["base"], sources["selected"], sources["image_ids"])
    protocol.exclusive_bytes(protocol.OUTPUT, payload)
    completion = {
        "schema_version": "maxim274-selective-fusion-byte-preserve-completion-v1.3",
        "freeze_sha256": expected_freeze,
        "independent_audit_sha256": expected_audit,
        "v1_1_completion_sha256": protocol.PINS["v1_1_completion"][1],
        "v1_1_solver_sha256": protocol.PINS["v1_1_solver"][1],
        "base251_solver_sha256": protocol.PINS["base251_solver"][1],
        "base251_image97_judge_sha256": protocol.PINS["base251_image97_judge"][1],
        "candidate_completion_sha256": protocol.PINS["candidate_completion"][1],
        "candidate_predictions_sha256": protocol.PINS["candidate_predictions"][1],
        "output_sha256": protocol.sha256_file(protocol.OUTPUT),
        "rows": 274,
        **census,
        "identity_used_for_branch_selection": False,
        "identity_used_postdecision_for_alignment": True,
        "gold_opened_by_compositor": False,
        "outcomes_opened_by_compositor": False,
        "semantic_tunables": 0,
    }
    if set(completion) != protocol.COMPLETION_KEYS:
        raise protocol.ProtocolError("internal completion keyset mismatch")
    protocol.exclusive_bytes(protocol.COMPLETION, protocol.canonical_json(completion))
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.expected_freeze_sha256, args.expected_independent_audit_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

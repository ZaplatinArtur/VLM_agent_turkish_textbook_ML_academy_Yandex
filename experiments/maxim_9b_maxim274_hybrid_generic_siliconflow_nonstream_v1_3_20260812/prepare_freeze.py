"""Derive an ID-free 256-row queue and freeze before any provider or score access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from generic_candidate import MAX_TOKENS, MODEL_ID, PROVIDER_NAME, PROVIDER_QUANTIZATION, REASONING, build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes
from nonstream_protocol import MAX_ATTEMPTS, RETRYABLE_KINDS, TIMEOUT_SECONDS, WORKERS, exclusive_bytes, exclusive_json, sha256_file, stable_bytes, utc_now

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
HYBRID = EXPERIMENTS / "maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812"
SOURCE_QUEUE = HYBRID / "runs" / "maxim274" / "generic_queue.jsonl"
HYBRID_FREEZE = HYBRID / "HYBRID_RULE_FREEZE.json"
ADAPTER = EXPERIMENTS / "maxim_9b_maxim274_generic_content_adapter_v1_20260812"
REJECTION = ADAPTER / "REJECTION_DO_NOT_RUN.json"
INVENTORY = ADAPTER / "frozen" / "INPUT_INVENTORY.json"
V6_2 = EXPERIMENTS / "maxim_9b_ykslop_generic_reasoning_nonstream_alt_v6_2_siliconflow_dev_20260812" / "DEV_EXECUTION_FREEZE.json"
BENCHMARK = HERE.parent.parent / "artifacts" / "baselines" / "basic_page_rag_v1" / "validation_274.jsonl"
BASELINE_JUDGE = HERE.parent.parent / "artifacts" / "baselines" / "basic_page_rag_v1" / "agent_rag_judge.jsonl"
STANDARD_SCORER = HERE.parent.parent / "scripts" / "score_maxim_full274.py"

QUEUE = HERE / "frozen" / "queue_content_only_256.jsonl"
ALIGNMENT = HERE / "frozen" / "outer_alignment_256.jsonl"
SMOKE = HERE / "frozen" / "smoke_request.json"
AUTH = HERE / "USER_AUTHORIZATION.json"
FREEZE = HERE / "EXECUTION_FREEZE.json"
SIDECAR = HERE / "EXECUTION_FREEZE_SHA256.txt"
PRIVATE = HERE / "PRIVATE_SCORE_FREEZE.json"
PRIVATE_SIDECAR = HERE / "PRIVATE_SCORE_FREEZE_SHA256.txt"

SOURCE_QUEUE_SHA = "b222a2fbc17afd33141802b727e302052b352743753221e44202a2bb5e156820"
HYBRID_FREEZE_SHA = "c904f1ea7151513cb83757cc80e21e8dd1cdbd8c7eb4fbf47a40ee40e35ac177"
REJECTION_SHA = "0951522485895c12bf9fee79029172487cfce4db90f67309d4a0ff4d5580b580"
INVENTORY_SHA = "c005d68fce5dafd9cc9897d0848b6a2629799d7eb301e8bacfee803956a32b2b"
V6_2_SHA = "c1b275c985489a8fd1e534d93138c00c288c91b22c735793bc5b7c24b726f099"
BENCHMARK_SHA = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
BASELINE_JUDGE_SHA = "59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4"
SCORER_SHA = "bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf"
SUPERSEDED_V1_FREEZE_SHA = "4a917b6442c8fcec81a37ec0ae99c7f8dbf9f4e57e7efbdb36ab420a8bdf1b6c"
SUPERSEDED_V1_1_FREEZE_SHA = "a3c43ae320f0a25cbc0030ee788e17accbeee1f3576af943fea05a01e288789f"
SUPERSEDED_V1_2_FREEZE_SHA = "8659377d0f93cc9110646d677734debe92fcaf37ffc46d1b80417883561df5d4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stable_bytes(path).decode("utf-8").splitlines() if line.strip()]


def desc(path: Path, rows: int | None = None) -> dict[str, Any]:
    value = {"path": Path(os.path.relpath(path, HERE)).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
    if rows is not None:
        value["rows"] = rows
    return value


def main() -> None:
    outputs = (QUEUE, ALIGNMENT, SMOKE, AUTH, FREEZE, SIDECAR, PRIVATE, PRIVATE_SIDECAR)
    if any(path.exists() for path in outputs):
        raise RuntimeError("freeze output exists; overwrite refused")
    implementations = {name: HERE / name for name in ("generic_candidate.py", "nonstream_protocol.py", "run_candidate.py", "smoke_protocol.py", "compose_hybrid.py", "score_private.py", "test_candidate.py")}
    if any(not path.is_file() for path in implementations.values()):
        raise RuntimeError("implementation incomplete")
    pins = ((SOURCE_QUEUE, SOURCE_QUEUE_SHA), (HYBRID_FREEZE, HYBRID_FREEZE_SHA), (REJECTION, REJECTION_SHA), (INVENTORY, INVENTORY_SHA), (V6_2, V6_2_SHA), (STANDARD_SCORER, SCORER_SHA))
    if any(sha256_file(path) != digest for path, digest in pins):
        raise RuntimeError("public/private-descriptor ancestry pin mismatch")
    rejection = json.loads(stable_bytes(REJECTION))
    if rejection.get("status") != "REJECTED_DO_NOT_RUN" or rejection.get("adapter_v1_live_api_calls") != 0:
        raise RuntimeError("rejected V1 lineage mismatch")
    source = read_jsonl(SOURCE_QUEUE)
    if len(source) != 256:
        raise RuntimeError("Hybrid V3.1 abstention queue denominator mismatch")
    public: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for row in source:
        if type(row.get("controller_id")) is not str or type(row.get("ocr_text")) is not str or not row["ocr_text"].strip():
            raise RuntimeError("source queue row mismatch")
        mode = "text_only" if row.get("input_mode") == "text_only" else "multimodal_degraded_to_ocr_only"
        content = {"schema_version": "maxim256-idfree-ocr-row-v1", "subject": row["subject"], "answer_type": row["answer_type"], "ocr_text": row["ocr_text"], "source_input_mode": mode}
        build_request(content)
        public.append(content)
        alignment.append({"schema_version": "maxim256-outer-alignment-v1", "task_id": row["controller_id"]})
    if sum(row["source_input_mode"] == "text_only" for row in public) != 70 or sum(row["source_input_mode"] == "multimodal_degraded_to_ocr_only" for row in public) != 186:
        raise RuntimeError("OCR-only coverage census mismatch")
    queue_bytes = b"".join(canonical_json_bytes(row) for row in public)
    alignment_bytes = b"".join(canonical_json_bytes(row) for row in alignment)
    exclusive_bytes(QUEUE, queue_bytes)
    exclusive_bytes(ALIGNMENT, alignment_bytes)
    smoke, _ = build_request(fixed_smoke_row())
    exclusive_bytes(SMOKE, canonical_json_bytes(smoke))
    implementation_sha = {name: sha256_file(path) for name, path in implementations.items()}
    # Authorization binds the implementation and inputs before freeze; freeze hash is filled
    # through a detached scope hash to avoid a self-reference cycle.
    auth = {"schema_version": "maxim256-openrouter-authorization-v1", "created_utc": utc_now(), "authorized": True, "user_statement": "User authorized autonomous OpenRouter Qwen3.5-9B evaluation with memory-only credential.", "authorized_scope": "one fixed non-benchmark smoke with up to three exact retries, then 256 Hybrid V3.1 abstained public rows with up to three exact retries", "queue_sha256": sha256_file(QUEUE), "alignment_sha256": sha256_file(ALIGNMENT), "implementation_sha256": implementation_sha, "model_id": MODEL_ID, "provider": PROVIDER_NAME, "provider_quantization": PROVIDER_QUANTIZATION, "api_key_storage": "interactive_memory_only_not_persisted"}
    exclusive_json(AUTH, auth)
    freeze = {
        "schema_version": "maxim256-hybrid-generic-siliconflow-freeze-v1", "state": "frozen_unexecuted_unscored", "created_utc": utc_now(), "rows": 256,
        "scope": "Hybrid V3.1 Maxim274 abstained rows only; outer selector already frozen, generic wire sees ID-free content rows",
        "model_id": MODEL_ID, "provider": PROVIDER_NAME, "provider_quantization": PROVIDER_QUANTIZATION, "hosted_bitwise_determinism_claimed": False,
        "workers": WORKERS, "timeout_seconds": TIMEOUT_SECONDS, "max_attempts": MAX_ATTEMPTS, "retryable_kinds": sorted(RETRYABLE_KINDS),
        "request_contract": {"stream": False, "seed_present": False, "reasoning": REASONING, "max_tokens": MAX_TOKENS, "temperature": 0.0, "top_p": 1.0, "provider_only": ["siliconflow"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["fp8"], "data_collection": "deny", "zdr": True, "strict_response_schema": True},
        "image_contract": {"decision": "ocr_only_fail_closed", "source_text_only": 70, "source_multimodal_degraded_to_ocr_only": 186, "image_bytes_sent": False, "all_rows_have_nonempty_frozen_ocr": True, "reason": "exact SiliconFlow endpoint catalog record lacks endpoint-specific input modality attestation"},
        "pre_outcome_contract": {"primary_hybrid_success_correct_at_least": 240, "total": 274, "accuracy_at_least": 240 / 274, "full_denominator": True, "frozen_before_smoke_or_live": True, "selection": "sole candidate; compose certified no-ID result on 18 accepted rows and this generic result on exactly 256 abstained rows", "failure_rows_count_wrong": True},
        "ancestry": {"hybrid_v3_1_freeze_sha256": HYBRID_FREEZE_SHA, "hybrid_generic_queue_sha256": SOURCE_QUEUE_SHA, "rejected_adapter_v1_sha256": REJECTION_SHA, "rejected_adapter_v1_live_calls": 0, "transport_pattern_v6_2_freeze_sha256": V6_2_SHA, "rejected_v1_not_imported_or_runnable": True, "supersedes_pre_live_candidate_v1_freeze_sha256": SUPERSEDED_V1_FREEZE_SHA, "supersedes_pre_live_candidate_v1_1_freeze_sha256": SUPERSEDED_V1_1_FREEZE_SHA, "supersedes_pre_live_candidate_v1_2_freeze_sha256": SUPERSEDED_V1_2_FREEZE_SHA, "supersession_reason": "V1 lacked fresh modality evidence; V1.1 needed scorer/resume hardening; V1.2 did not bind standard score provenance to the exact composed solver; all predecessors had zero provider calls, outputs, scores or PASS audits"},
        "artifacts": {"queue": desc(QUEUE, 256), "alignment": desc(ALIGNMENT, 256), "smoke_request": desc(SMOKE), "authorization": desc(AUTH), "provider_snapshot": desc(HERE / "frozen" / "provider_endpoint_snapshot.json"), "zdr_snapshot": desc(HERE / "frozen" / "zdr_inventory_snapshot.json"), "image_capability": desc(HERE / "frozen" / "IMAGE_CAPABILITY_EVIDENCE.json"), "generic_output_contract": desc(HERE / "frozen" / "GENERIC_OUTPUT_CONTRACT.json")},
        "implementation": {name.rsplit(".", 1)[0]: desc(path) for name, path in implementations.items()},
        "guards": {"api_called": False, "gold_opened": False, "outcomes_opened": False, "final_opened": False, "wire_identity_or_hash": False, "outer_alignment_used_by_selector": False, "credential_persisted": False, "independent_pass_audit_required": True},
    }
    exclusive_json(FREEZE, freeze)
    freeze_sha = sha256_file(FREEZE)
    exclusive_bytes(SIDECAR, (freeze_sha + "\n").encode())
    # Runtime compares this authorization placeholder against the immutable freeze
    # artifact descriptor, not a circular embedded freeze hash.
    private = {"schema_version": "maxim256-hybrid-private-score-freeze-v1", "state": "private_score_frozen_unexecuted", "created_utc": utc_now(), "execution_freeze_sha256": freeze_sha, "artifacts": {"benchmark": {"path": Path(os.path.relpath(BENCHMARK, HERE)).as_posix(), "rows": 274, "sha256": BENCHMARK_SHA, "size": BENCHMARK.stat().st_size}, "baseline_judge": {"path": Path(os.path.relpath(BASELINE_JUDGE, HERE)).as_posix(), "rows": 274, "sha256": BASELINE_JUDGE_SHA, "size": BASELINE_JUDGE.stat().st_size}, "standard_scorer": desc(STANDARD_SCORER), "composer": desc(HERE / "compose_hybrid.py"), "aggregate_wrapper": desc(HERE / "score_private.py")}, "scoring_contract": {"denominator": 274, "success_correct_at_least": 240, "success_accuracy_at_least": 240 / 274, "missing_malformed_or_error": "wrong", "exact_composition_recomputed_from_route_decisions_predictions_completion_and_outer_alignment": True, "persisted_solver_must_be_byte_identical_to_recomposition": True, "exact_composed_solver_sha_required_in_standard_score_provenance": True, "exact_externally_pinned_image_judge_sha_required_in_standard_score_provenance": True, "score_json_must_match_both_exact_solver_and_image_judge_inputs": True, "aggregate_only_wrapper": True, "task_ids_in_private_result": False, "gold_answers_in_private_result": False, "per_row_outcomes_in_private_result": False}, "guards": {"benchmark_not_opened_during_freeze": True, "baseline_judge_not_opened_during_freeze": True, "gold_or_outcomes_not_opened_during_freeze": True, "score_only_after_atomic_completion_and_composition": True}}
    exclusive_json(PRIVATE, private)
    private_sha = sha256_file(PRIVATE)
    exclusive_bytes(PRIVATE_SIDECAR, (private_sha + "\n").encode())
    print(json.dumps({"freeze_sha256": freeze_sha, "authorization_sha256": sha256_file(AUTH), "private_score_freeze_sha256": private_sha, "queue_sha256": sha256_file(QUEUE), "alignment_sha256": sha256_file(ALIGNMENT)}, indent=2))


if __name__ == "__main__":
    main()

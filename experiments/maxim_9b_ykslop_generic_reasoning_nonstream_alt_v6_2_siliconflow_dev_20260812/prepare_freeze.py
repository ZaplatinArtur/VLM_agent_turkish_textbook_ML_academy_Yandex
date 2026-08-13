"""Freeze the non-streaming SiliconFlow V6 DEV protocol without API or gold access."""

from __future__ import annotations

import json
from pathlib import Path

from generic_candidate import MAX_TOKENS, MODEL_ID, PROVIDER_NAME, PROVIDER_QUANTIZATION, REASONING, build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes
from nonstream_protocol import MAX_ATTEMPTS, RETRYABLE_KINDS, TIMEOUT_SECONDS, WORKERS, exclusive_bytes, exclusive_json, sha256_file, stable_bytes, utc_now

HERE = Path(__file__).resolve().parent
QUEUE = HERE / "frozen" / "queue_public_content_only.jsonl"
PROVIDER = HERE / "frozen" / "provider_endpoint_snapshot.json"
ZDR = HERE / "frozen" / "zdr_inventory_snapshot.json"
SMOKE = HERE / "frozen" / "smoke_request.json"
AUTH = HERE / "USER_AUTHORIZATION.json"
FREEZE = HERE / "DEV_EXECUTION_FREEZE.json"
SIDECAR = HERE / "DEV_EXECUTION_FREEZE_SHA256.txt"
V3_QUEUE = HERE.parent / "maxim_9b_ykslop_generic_reasoning_sse_alt_v3_dev_20260812" / "frozen" / "queue_public_content_only.jsonl"
V5_RESULT = HERE.parent / "maxim_9b_ykslop_generic_reasoning_sse_alt_v5_siliconflow_smoke_20260812" / "PROTOCOL_SMOKE_RESULT.json"

QUEUE_SHA = "b2b03dfb53218e7e099c5129c4ef6acd096d8c44e44b2e82a37c84c746e549a2"
PROMPT_SHA = "844e11d2007f88ca6732a00b7a87a34da200c730f52327f1f0c7c3819998061f"
PROVIDER_SHA = "020df7beb073ecdb785e2b96651b9068d911fed8ba72e5895aa32237be824c12"
ZDR_SHA = "e2bab064b51183f88535f18c88dc820d71987026c7a757f58c2b2bfa6a9755a2"
V5_FAILED_SHA = "238cb07a7d7729e743236f6c2eb82fa66033ed8af8f020dbbf320577c0acaa4a"
SOURCE_PUBLIC_SHA = "eebfda230a10ef98f07c53b5d7ab55cca24a718c8439074192d4f29156acc47c"
SPLIT_SHA = "6fb8474a9e0e71c2afae3c9bef20f22cbc6a5b7ae8928321b2d919afcfd32f9e"
SUPERSEDED_V6_FREEZE_SHA = "61d49ac92cfc15ee3b74bcc3c09e7ec01ef06da14853e3fd91951b382c989d2b"
SUPERSEDED_V6_1_FREEZE_SHA = "dffd172b48e0c6b8eecaa30636b0b08cf2a23c48bba86b136f12e6fa4391ec22"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in stable_bytes(path).decode("utf-8").splitlines() if line.strip()]


def desc(path: Path, count: int | None = None) -> dict:
    value = {"path": path.relative_to(HERE).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
    if count is not None:
        value["rows"] = count
    return value


def main() -> None:
    if any(path.exists() for path in (SMOKE, AUTH, FREEZE, SIDECAR)):
        raise RuntimeError("freeze output exists; overwrite refused")
    required = [HERE / name for name in ("generic_candidate.py", "nonstream_protocol.py", "run_dev.py", "smoke_protocol.py", "test_v6_nonstream.py", "INDEPENDENT_AUDIT_TEMPLATE.json")]
    if any(not path.is_file() for path in required):
        raise RuntimeError("pre-freeze implementation incomplete")
    if sha256_file(QUEUE) != QUEUE_SHA or stable_bytes(QUEUE) != stable_bytes(V3_QUEUE):
        raise RuntimeError("185 queue is not byte-identical to frozen direct V3")
    public = rows(QUEUE)
    if len(public) != 185:
        raise RuntimeError("full denominator mismatch")
    requests = [build_request(row)[0] for row in public]
    prompt_sha = sha256_bytes(canonical_json_bytes([request["messages"] for request in requests]))
    if prompt_sha != PROMPT_SHA:
        raise RuntimeError("prompt sequence differs from direct V3")
    if sha256_file(PROVIDER) != PROVIDER_SHA or sha256_file(ZDR) != ZDR_SHA:
        raise RuntimeError("current provider/ZDR evidence pin mismatch")
    endpoint = json.loads(stable_bytes(PROVIDER))["endpoint"]
    if endpoint.get("provider_name") != "SiliconFlow" or endpoint.get("quantization") != "fp8" or "seed" in endpoint.get("supported_parameters", []):
        raise RuntimeError("SiliconFlow FP8/no-seed evidence mismatch")
    zdr = json.loads(stable_bytes(ZDR))
    if zdr.get("exact_siliconflow_fp8_present") is not True:
        raise RuntimeError("SiliconFlow FP8 is absent from frozen ZDR inventory")
    if sha256_file(V5_RESULT) != V5_FAILED_SHA:
        raise RuntimeError("failed V5 SSE lineage mismatch")
    smoke_request, _ = build_request(fixed_smoke_row())
    exclusive_bytes(SMOKE, canonical_json_bytes(smoke_request))
    impl_paths = {"candidate": HERE / "generic_candidate.py", "protocol": HERE / "nonstream_protocol.py", "runner": HERE / "run_dev.py", "smoke": HERE / "smoke_protocol.py"}
    impl_sha = {key: sha256_file(path) for key, path in impl_paths.items()}
    auth = {
        "schema_version": "generic-medium-nonstream-user-authorization-v6",
        "created_utc": utc_now(),
        "user_statement": "User explicitly authorized autonomous OpenRouter Qwen3.5-9B evaluation; API key remains memory-only and is never persisted.",
        "authorized_openrouter_calls": True,
        "authorized_scope": "185 public DEV calls plus up to three exact-retry fixed non-benchmark protocol smoke attempts",
        "queue_sha256": QUEUE_SHA,
        "implementation_sha256": impl_sha,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "api_key_storage": "interactive_memory_only_not_persisted",
        "live_gate": "external independent PASS audit SHA required",
    }
    exclusive_json(AUTH, auth)
    freeze = {
        "schema_version": "generic-medium-reasoning-nonstream-dev-freeze-v6",
        "state": "frozen_unexecuted_unscored",
        "created_utc": utc_now(),
        "scope": "full_185_public_DEV_content_only_with_frozen_general_theory",
        "row_count": 185,
        "source_public_rows": 185,
        "development_split_sha256": SPLIT_SHA,
        "source_public_sha256": SOURCE_PUBLIC_SHA,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "hosted_observation": "nondeterministic_no_seed_supported",
        "workers": WORKERS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "retryable_kinds": sorted(RETRYABLE_KINDS),
        "failed_v5_sse_lineage": {"path": str(V5_RESULT.relative_to(HERE.parent)).replace("\\", "/"), "sha256": V5_FAILED_SHA, "error_kind": "sse_schema"},
        "supersedes_pre_live_v6_freeze_sha256": SUPERSEDED_V6_FREEZE_SHA,
        "supersession_reason": "V6 incorrectly rejected legitimate medium-reasoning usage token accounting; no provider call or output existed",
        "supersedes_pre_live_v6_1_freeze_sha256": SUPERSEDED_V6_1_FREEZE_SHA,
        "v6_1_supersession_reason": "V6.1 could persist raw provider or exception detail; no provider call or output existed",
        "pre_outcome_success_criterion": {"correct_at_least": 148, "total": 185, "accuracy_at_least": 0.8, "full_denominator": True, "frozen_before_smoke_or_live": True},
        "selection_contract": "sole SiliconFlow candidate; no adaptive arm choice",
        "request_contract": {"stream": False, "seed_present": False, "reasoning": REASONING, "max_tokens": MAX_TOKENS, "temperature": 0.0, "top_p": 1.0, "provider_only": ["siliconflow"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["fp8"], "data_collection": "deny", "zdr": True, "response": "strict JSON object with one A-E answer field", "choice_order": "content-derived permutation mapped back locally"},
        "ancestry": {"byte_identical_direct_v3_queue_sha256": QUEUE_SHA, "prompt_sequence_identical_direct_v3_sha256": prompt_sha, "opaque_ids_removed_before_runtime": True, "outcome_selection": False},
        "artifacts": {"queue": desc(QUEUE, 185), "provider_snapshot": desc(PROVIDER), "zdr_snapshot": desc(ZDR), "smoke_request": desc(SMOKE), "authorization": desc(AUTH)},
        "implementation": {key: desc(path) for key, path in impl_paths.items()},
        "verification": {"tests": desc(HERE / "test_v6_nonstream.py"), "independent_audit_template": desc(HERE / "INDEPENDENT_AUDIT_TEMPLATE.json"), "prepare_script": desc(Path(__file__))},
        "guards": {"gold_read_by_runtime": False, "final_read_by_runtime": False, "prior_outcomes_read_by_runtime": False, "runtime_opaque_ids": False, "api_key_persisted": False, "live_requires_independent_audit": True, "provider_calls_before_freeze": 0},
    }
    exclusive_json(FREEZE, freeze)
    digest = sha256_file(FREEZE)
    exclusive_bytes(SIDECAR, (digest + "\n").encode())
    print(json.dumps({"freeze_sha256": digest, "authorization_sha256": sha256_file(AUTH), "queue_sha256": QUEUE_SHA, "prompt_sequence_sha256": prompt_sha}, indent=2))


if __name__ == "__main__":
    main()

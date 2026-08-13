"""Immutable byte-preserving successor to audited selective fusion V1.1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
V11 = EXPERIMENTS / "maxim_9b_maxim274_identity_free_selective_fusion_v1_20260812"
BASE251_ROOT = EXPERIMENTS / "maxim_9b_content_source_router_noid_v1_20260812"
CANDIDATE = EXPERIMENTS / "maxim_9b_maxim274_hybrid_generic_siliconflow_nonstream_v1_3_20260812"
V16 = EXPERIMENTS / "maxim_9b_maxim274_hybrid_image97_judge_together_v1_6_20260812"
V12 = EXPERIMENTS / "maxim_9b_maxim274_identity_free_selective_fusion_byte_preserve_v1_2_20260812"

V11_FREEZE = V11 / "FUSION_RULE_FREEZE.json"
V11_AUDIT = V11 / "INDEPENDENT_AUDIT.json"
V11_COMPLETION = V11 / "FUSION_COMPLETION.json"
V11_SOLVER = V11 / "runs/selective_fusion_solver_274.jsonl"
V11_IMPLEMENTATION = V11 / "selective_fusion.py"
BASE251_FREEZE = BASE251_ROOT / "output/FREEZE.json"
BASE251_AUDIT = BASE251_ROOT / "INDEPENDENT_AUDIT.json"
BASE251_SOLVER = BASE251_ROOT / "output/frozen/arms/strict_b_over_base240/solver.jsonl"
BASE251_IMAGE_JUDGE = BASE251_ROOT / "output/frozen/arms/strict_b_over_base240/image97_judge.jsonl"
CANDIDATE_COMPLETION = CANDIDATE / "COMPLETION.json"
CANDIDATE_PREDICTIONS = CANDIDATE / "runs/generic_predictions_256.jsonl"
IMAGE97_ALIGNMENT = V16 / "frozen/outer_alignment_97.jsonl"
CANDIDATE_TEXT_ADAPTER = REPO / "scripts/prepare_canonical_judge97_input.py"
SCORER = REPO / "scripts/score_maxim_full274.py"
BENCHMARK = REPO / "artifacts/baselines/basic_page_rag_v1/validation_274.jsonl"
SCORER_BASELINE_JUDGE = REPO / "artifacts/baselines/basic_page_rag_v1/agent_rag_judge.jsonl"

V12_FREEZE = V12 / "BYTE_PRESERVE_RULE_FREEZE.json"
FREEZE = HERE / "BYTE_PRESERVE_RULE_FREEZE_V1_3.json"
FREEZE_SHA_FILE = HERE / "BYTE_PRESERVE_RULE_FREEZE_V1_3_SHA256.txt"
AUDIT = HERE / "INDEPENDENT_AUDIT.json"
OUTPUT = HERE / "runs/selective_fusion_byte_preserve_v1_3_solver_274.jsonl"
COMPLETION = HERE / "BYTE_PRESERVE_COMPLETION_V1_3.json"
PRIVATE_RESULT = HERE / "PRIVATE_RESULT_V1_3.json"

PINS = {
    "v1_1_freeze": (V11_FREEZE, "a384e8d7e975b5f93b655f2af671004dde3e742895569bba81e76567d22175e1"),
    "v1_1_audit": (V11_AUDIT, "265d44355c8340b2eb78a2da51000ce2358b7a7f4badbc2294f211f949a101d6"),
    "v1_1_completion": (V11_COMPLETION, "7d0fcfaa36a24aae07a57f7fb20b91ae8708672259eb61dfb316680b348b2b86"),
    "v1_1_solver": (V11_SOLVER, "fc3cd33e3b439f902a2241f50a6d291b3f86e00d3384f55b512b8f8b99ec7e45"),
    "v1_1_implementation": (V11_IMPLEMENTATION, "abd56bf56514c97dd131371e801d9f3c258647c8c5496dd392667885d68d6bae"),
    "base251_freeze": (BASE251_FREEZE, "76a09995b1104b4b5fec67bb737e73e4a5b21032916f37a24a563118802a8a7c"),
    "base251_audit": (BASE251_AUDIT, "c1d67ea44fa88f487aa4e65c7c78c3b1e13cbce0564ac1b06f50d009a4e45d82"),
    "base251_solver": (BASE251_SOLVER, "f87f6ad41817c3d55fde5630781cd6f9f958350bfde72bdebeb8567b454c832a"),
    "base251_image97_judge": (BASE251_IMAGE_JUDGE, "f2375bf3cea7492e3947ea285ca5db9262a50a5c5abfc92f2dc68cc1386095bf"),
    "candidate_completion": (CANDIDATE_COMPLETION, "ba031a65a60b1235183570938afa9c11d85429927be994a3546b45095386be11"),
    "candidate_predictions": (CANDIDATE_PREDICTIONS, "a477f40e215d708adafc93f201af2142b7ae3915bb0dbabb00c8d1aaf33d4fa4"),
    "image97_alignment": (IMAGE97_ALIGNMENT, "b377e2bec4fb39b9f43d665e8a9d22131c37199f2f2d0d920ff92caae6627f7d"),
    "candidate_text_adapter": (CANDIDATE_TEXT_ADAPTER, "a0d670401f8165360504f5f24b81ce25c9626f6ed01c7f7a796f6e27549ad74a"),
    "official_scorer": (SCORER, "bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf"),
    "benchmark": (BENCHMARK, "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"),
    "scorer_baseline_judge": (SCORER_BASELINE_JUDGE, "59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4"),
    "withheld_v1_2_freeze": (V12_FREEZE, "904837c402a382a66d6604671791abb67c7865af914f7f1b1c2b8604423e0e8e"),
}

HEX64 = re.compile(r"[0-9a-f]{64}")
BASELINE_BRANCHES = {"baseline251", "certified_noid_baseline251"}
GENERIC_BRANCH = "generic_valid_over_unparseable_baseline"
SUCCESS_THRESHOLD_CORRECT = 250
IMPLEMENTATION_FILES = {
    "README.md",
    "protocol.py",
    "compose.py",
    "prepare_freeze.py",
    "score_reusing_base251_judge.py",
    "test_byte_preserve.py",
    "INDEPENDENT_AUDIT_TEMPLATE.json",
}
AUDIT_CHECKS = {
    "freeze_closure_and_runtime_self_hashes",
    "v1_1_freeze_audit_result_exact_pins",
    "candidate_atomic_completion_and_predictions_exact_pins",
    "base251_solver_and_image97_judge_exact_pins",
    "zero_tunable_exact_v1_1_decision_import",
    "baseline_selected_rows_copy_raw_base251_bytes",
    "generic_selected_rows_copy_raw_v1_1_bytes",
    "two_generic_switches_outside_image97",
    "all_97_image_rows_objects_and_candidate_text_utf8_exact",
    "selector_identity_free_and_identity_alignment_only",
    "no_gold_outcome_or_judge_verdict_read_by_compositor",
    "postscore_builder_exposure_and_v1_2_withhold_disclosed",
    "scorer_recomputes_exact_payload_and_validates_completion",
    "private_scorer_fixed_paths_provenance_and_threshold_250",
    "hostile_deterministic_and_implementation_mutations_rejected",
    "tests_pass",
}
COMPLETION_KEYS = {
    "schema_version",
    "freeze_sha256",
    "independent_audit_sha256",
    "v1_1_completion_sha256",
    "v1_1_solver_sha256",
    "base251_solver_sha256",
    "base251_image97_judge_sha256",
    "candidate_completion_sha256",
    "candidate_predictions_sha256",
    "output_sha256",
    "rows",
    "baseline_rows_copied_byte_exact",
    "generic_rows_copied_from_v1_1_byte_exact",
    "image97_rows_base251_byte_and_object_exact",
    "image97_candidate_text_utf8_exact",
    "identity_used_for_branch_selection",
    "identity_used_postdecision_for_alignment",
    "gold_opened_by_compositor",
    "outcomes_opened_by_compositor",
    "semantic_tunables",
}


class ProtocolError(RuntimeError):
    pass


def stable_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ProtocolError(f"required file missing: {path}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or len(data) != before.st_size:
        raise ProtocolError(f"file changed during read: {path}")
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(stable_bytes(path))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(stable_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ProtocolError(f"JSON root is not an object: {path}")
    return value


def jsonl_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    result: list[tuple[dict[str, Any], bytes]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(keepends=True), 1):
        if not raw.strip():
            continue
        if not raw.endswith(b"\n"):
            raise ProtocolError(f"row lacks LF terminator: {path}:{number}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid JSONL: {path}:{number}") from exc
        if type(row) is not dict:
            raise ProtocolError(f"non-object JSONL row: {path}:{number}")
        result.append((row, raw))
    return result


def ordered_ids(rows: Iterable[tuple[dict[str, Any], bytes]], label: str) -> list[str]:
    values = [row.get("task_id") for row, _ in rows]
    if any(type(value) is not str or not value for value in values) or len(values) != len(set(values)):
        raise ProtocolError(f"{label} identity closure failed")
    return values  # type: ignore[return-value]


def verify_pins() -> None:
    for name, (path, expected) in PINS.items():
        # Hashing is byte provenance, not semantic parsing. In particular the
        # two judge artifacts are never opened as JSON by this compositor.
        if sha256_file(path) != expected:
            raise ProtocolError(f"pin mismatch: {name}")


def descriptor(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": os.path.relpath(path.resolve(), HERE.resolve()).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _verify_descriptor(value: Any, expected_path: Path, label: str) -> None:
    if type(value) is not dict or set(value) not in ({"path", "sha256", "size"}, {"path", "sha256", "size", "rows"}):
        raise ProtocolError(f"invalid frozen descriptor: {label}")
    path = (HERE / str(value["path"])).resolve()
    if path != expected_path.resolve() or path.stat().st_size != value["size"] or sha256_file(path) != value["sha256"]:
        raise ProtocolError(f"frozen descriptor mismatch: {label}")


def verify_own_protocol(expected_freeze: str, expected_audit: str) -> dict[str, Any]:
    """Bind runtime code and every fixed input to the independently audited freeze."""
    if HEX64.fullmatch(expected_freeze) is None or HEX64.fullmatch(expected_audit) is None:
        raise ProtocolError("freeze/audit arguments must be lowercase SHA256")
    if sha256_file(FREEZE) != expected_freeze or FREEZE_SHA_FILE.read_text(encoding="ascii").strip() != expected_freeze:
        raise ProtocolError("own freeze pin mismatch")
    if sha256_file(AUDIT) != expected_audit:
        raise ProtocolError("own audit pin mismatch")
    freeze, audit = read_json(FREEZE), read_json(AUDIT)
    if (
        freeze.get("schema_version") != "maxim274-selective-fusion-byte-preserve-freeze-v1.3"
        or freeze.get("state") != "frozen_after_v1_1_atomic_completion_before_v1_3_output_and_private_score"
        or audit.get("schema_version") != "maxim274-selective-fusion-byte-preserve-independent-audit-v1.3"
        or audit.get("status") != "PASS"
        or audit.get("freeze_sha256") != expected_freeze
        or type(audit.get("checks")) is not dict
        or set(audit["checks"]) != AUDIT_CHECKS
        or any(value is not True for value in audit["checks"].values())
        or audit.get("guards") != {
            "api_called": False,
            "private_score_executed": False,
            "judge_verdict_content_opened": False,
            "gold_opened": False,
        }
    ):
        raise ProtocolError("own freeze/audit semantic mismatch")
    artifacts = freeze.get("artifacts")
    implementation = freeze.get("implementation")
    if type(artifacts) is not dict or set(artifacts) != set(PINS) or type(implementation) is not dict or set(implementation) != IMPLEMENTATION_FILES:
        raise ProtocolError("own descriptor keyset mismatch")
    for name, (path, _) in PINS.items():
        _verify_descriptor(artifacts[name], path, name)
    for name in IMPLEMENTATION_FILES:
        _verify_descriptor(implementation[name], HERE / name, name)
    return freeze


def exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ProtocolError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if stable_bytes(path) != data:
        raise ProtocolError("post-write byte mismatch")


def branch(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if type(generation) is not dict or generation.get("identity_used_by_branch_selector") is not False:
        raise ProtocolError("V1.1 decision lacks identity-free provenance")
    value = generation.get("selected_branch")
    if value not in BASELINE_BRANCHES | {GENERIC_BRANCH}:
        raise ProtocolError("unknown V1.1 branch")
    return str(value)


def load_adapter() -> Any:
    if sha256_file(CANDIDATE_TEXT_ADAPTER) != PINS["candidate_text_adapter"][1]:
        raise ProtocolError("candidate-text adapter pin mismatch")
    spec = importlib.util.spec_from_file_location("byte_preserve_candidate_adapter", CANDIDATE_TEXT_ADAPTER)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot load candidate-text adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def threshold_met(correct: int) -> bool:
    return type(correct) is int and correct >= SUCCESS_THRESHOLD_CORRECT

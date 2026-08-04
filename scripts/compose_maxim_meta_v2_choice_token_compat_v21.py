#!/usr/bin/env python3
"""Compose the preregistered, gold-blind meta-v2 choice-token compatibility v2.1.

The only new behavior is generic: a ``choice`` verdict rejected solely because
its final answer is not A-E may be recovered when the saved raw verdict uses
exactly one ASCII digit and passes every other frozen meta-v2 validator and
selection gate.  All other error rows copy the frozen subject Router exactly.
Normal v2 rows are copied byte-for-byte at the row-content level.  This module
never opens labels, references, qrels, scores, or judge outputs.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import run_maxim_final_meta_verifier_v2 as v2
except ModuleNotFoundError:
    from scripts import run_maxim_final_meta_verifier_v2 as v2


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COMPAT_PROFILE = (
    REPO
    / "reports"
    / "maxim_final_meta_verifier_v2_choice_token_compat_v21_20260803"
    / "profile.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "reports"
    / "maxim_final_meta_verifier_v2_choice_token_compat_v21_20260803"
    / "run"
)

SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-composer-v21"
AUDIT_SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-audit-v21"
MANIFEST_SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-manifest-v21"
CONDITION = "maxim_final_meta_verifier_v2_choice_token_compat_v21"
PROMPT_VERSION = "original-first-anonymous-10way-verification-v2+choice-token-compat-v21"
EXPECTED_ROWS = 274
EXPECTED_PROFILE_SHA256 = "7f17d7be02e4e38835670ae2e52a41c0850bea8761c88c8e494bcb30e4dfbd52"
EXPECTED_QUEUE_SHA256 = "fbcc327b63cc354a2e87b2c92d64585ed62afca2e129bb328577dedefaf6d4f0"
EXPECTED_PREPARATION_MANIFEST_SHA256 = "0e066f523b7e59625e18913939cc102f1aefd9ace1fa522407e5e099922e64f1"
EXPECTED_V2_PROFILE_SHA256 = "9a5a9b721ef772f556bc9ec24bd39230ae116039f2f7e89abf941990d853468d"
EXPECTED_ROUTER_SHA256 = "34da8ef69619a8ba1f184cdfd1e6dcaf0fbdbdd1bfc50c711244a68f7d26a574"
STRICT_REJECTION = (
    "MetaVerifierError: choice final_answer must be exactly one letter A-E"
)
SEMANTIC_ATTEMPTS = 2
MIN_CONFIDENCE = 0.70
MIN_EVIDENCE = 2
_DIGIT = re.compile(r"[0-9]")
_FORBIDDEN_EVALUATION_KEYS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "gold_answer",
        "gold_label",
        "test_label",
        "qrels",
        "acceptable_answers",
        "judge_verdict",
        "judge_score",
    }
)


class CompatibilityError(ValueError):
    pass


def _assert_no_evaluation_keys(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if folded in _FORBIDDEN_EVALUATION_KEYS or "reference_answer" in folded:
                raise CompatibilityError(f"forbidden evaluation key at {location}.{key}")
            _assert_no_evaluation_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_evaluation_keys(child, f"{location}[{index}]")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CompatibilityError(f"{label}: expected JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CompatibilityError(f"{label}: cannot read JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompatibilityError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise CompatibilityError(f"{label}:{line_number}: row is not an object")
        values.append(value)
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_sha(row: Mapping[str, Any]) -> str:
    return v2.preparation.stable_sha256(dict(row))


def _assert_file_sha(path: Path, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise CompatibilityError(
            f"{label} SHA mismatch: expected={expected}, actual={actual}"
        )


def _index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise CompatibilityError(f"{label}: missing/duplicate task_id")
        result[task_id] = row
    return result


def validate_compat_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != "maxim-final-meta-choice-token-compat-profile-v21":
        raise CompatibilityError("compat profile schema mismatch")
    if profile.get("condition") != CONDITION:
        raise CompatibilityError("compat profile condition mismatch")
    if profile.get("gold_access") is not False:
        raise CompatibilityError("compat profile is not gold-blind")
    if profile.get("score_or_judge_inputs_allowed") is not False:
        raise CompatibilityError("compat profile permits score/judge inputs")
    if profile.get("selection_uses_task_id") is not False:
        raise CompatibilityError("compat profile permits task-specific selection")
    if profile.get("selection_uses_subject") is not False:
        raise CompatibilityError("compat profile permits subject-specific selection")
    binding = profile.get("frozen_v2_bindings")
    if not isinstance(binding, Mapping):
        raise CompatibilityError("compat profile bindings missing")
    if binding.get("strict_rejection") != STRICT_REJECTION:
        raise CompatibilityError("strict rejection binding mismatch")
    if binding.get("semantic_attempts") != SEMANTIC_ATTEMPTS:
        raise CompatibilityError("semantic-attempt binding mismatch")
    if float(binding.get("min_confidence") or -1) != MIN_CONFIDENCE:
        raise CompatibilityError("confidence binding mismatch")
    if int(binding.get("min_decisive_evidence") or -1) != MIN_EVIDENCE:
        raise CompatibilityError("evidence binding mismatch")
    if list(binding.get("candidate_ids") or []) != list(v2.preparation.OPAQUE_IDS):
        raise CompatibilityError("candidate ID binding mismatch")


def _failure_parts(error: Any) -> list[str]:
    if not isinstance(error, str) or not error.strip():
        return []
    return [value.strip() for value in error.split(" | ")]


def assess_numeric_compatibility(
    *, verifier_row: Mapping[str, Any], queue_row: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if str(queue_row.get("answer_type") or "").casefold() != "choice":
        reasons.append("answer_type_is_not_choice")
    parts = _failure_parts(verifier_row.get("error"))
    if len(parts) != SEMANTIC_ATTEMPTS:
        reasons.append("rejection_count_differs_from_frozen_semantic_attempts")
    if not parts or any(value != STRICT_REJECTION for value in parts):
        reasons.append("rejection_is_not_solely_strict_A_E")
    if verifier_row.get("semantic_attempt") != SEMANTIC_ATTEMPTS:
        reasons.append("semantic_attempt_record_mismatch")
    call = verifier_row.get("call")
    if not isinstance(call, Mapping):
        reasons.append("saved_call_missing")
    else:
        if call.get("parse_error") is not None:
            reasons.append("saved_call_has_parse_error")
        if call.get("recovered_partial") is not False:
            reasons.append("saved_call_is_partial")
        if call.get("finish_reason") != "stop":
            reasons.append("saved_call_finish_reason_is_not_stop")
    raw = verifier_row.get("raw_response")
    parsed: Any = None
    if not isinstance(raw, str) or not raw.strip():
        reasons.append("saved_raw_response_missing")
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            reasons.append("saved_raw_response_is_not_exact_JSON")
    if not isinstance(parsed, Mapping):
        reasons.append("saved_raw_response_is_not_an_object")
        return None, sorted(set(reasons))
    answer = str(parsed.get("final_answer") or "").strip()
    if not _DIGIT.fullmatch(answer):
        reasons.append("raw_final_answer_is_not_one_ASCII_digit")

    # This invokes the exact frozen validator for every field and semantic
    # invariant while changing only the one predicate extended by v2.1.
    sentinel = copy.deepcopy(dict(parsed))
    sentinel["final_answer"] = "A"
    try:
        validated = v2.validate_verdict(
            sentinel,
            candidate_ids=v2.preparation.OPAQUE_IDS,
            answer_type="choice",
        )
    except Exception as exc:
        reasons.append(f"other_frozen_validator_failed:{type(exc).__name__}:{exc}")
        return None, sorted(set(reasons))
    validated["final_answer"] = answer
    if validated.get("abstain") is not False:
        reasons.append("abstain_gate_failed")
    if validated.get("answer_format_verified") is not True:
        reasons.append("self_reported_format_gate_failed")
    if float(validated.get("confidence") or 0.0) < MIN_CONFIDENCE:
        reasons.append("confidence_gate_failed")
    evidence = validated.get("decisive_evidence")
    if not isinstance(evidence, list) or len(evidence) < MIN_EVIDENCE:
        reasons.append("evidence_gate_failed")
    result_for_policy = dict(verifier_row)
    result_for_policy["error"] = None
    result_for_policy["verdict"] = validated
    selection, selected_answer, policy_reason = v2.apply_frozen_policy(
        result_for_policy,
        min_confidence=MIN_CONFIDENCE,
        min_evidence=MIN_EVIDENCE,
    )
    if (
        selection != "meta_verifier"
        or selected_answer != answer
        or policy_reason != "valid_supported_meta_answer"
    ):
        reasons.append("frozen_selection_policy_failed")
    if reasons:
        return None, sorted(set(reasons))
    return validated, []


def _compose_recovered_solver(
    *,
    verifier_row: Mapping[str, Any],
    verdict: Mapping[str, Any],
    queue_sha256: str,
    compat_profile_sha256: str,
    source_verifier_sha256: str,
    source_v2_solver_sha256: str,
) -> dict[str, Any]:
    call = verifier_row.get("call")
    usage = call if isinstance(call, Mapping) else {}
    return {
        "task_id": str(verifier_row["task_id"]),
        "condition": CONDITION,
        "model": str(verifier_row.get("model") or ""),
        "prompt_version": PROMPT_VERSION,
        "final_answer": str(verdict["final_answer"]),
        "solution_steps": "\n".join(
            str(item) for item in verdict.get("decisive_evidence") or []
        ),
        "reasoning": str(verdict.get("independent_reasoning") or ""),
        "forced_answer": False,
        "raw_response": verifier_row.get("raw_response"),
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "enable_thinking": False,
            "structured_mode": "strict_response_format_choice_token_compat_v21",
            "gold_access": False,
            "original_question_and_images_primary": True,
            "candidate_source_identities_seen": False,
            "queue_sha256": queue_sha256,
            "queue_request_sha256": verifier_row["queue_request_sha256"],
            "verifier_request_sha256": verifier_row["verifier_request_sha256"],
            "confidence": verdict.get("confidence"),
            "answer_format_verified": verdict.get("answer_format_verified"),
            "selection_reason": "valid_supported_meta_answer_choice_token_compat_v21",
            "compat_profile_sha256": compat_profile_sha256,
            "source_verifier_row_sha256": source_verifier_sha256,
            "source_v2_solver_row_sha256": source_v2_solver_sha256,
            "source_rejection": STRICT_REJECTION,
        },
        "tool_calls": [],
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "latency_s": float(usage.get("latency_s") or 0.0),
        },
        "error": None,
    }


def compose_rows(
    *,
    queue: Sequence[Mapping[str, Any]],
    verifier: Sequence[Mapping[str, Any]],
    v2_solver: Sequence[Mapping[str, Any]],
    router: Sequence[Mapping[str, Any]],
    queue_sha256: str,
    compat_profile_sha256: str,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if not (
        len(queue)
        == len(verifier)
        == len(v2_solver)
        == len(router)
        == expected_rows
    ):
        raise CompatibilityError("source row counts differ")
    task_order = [str(row.get("task_id") or "") for row in queue]
    if not all(task_order) or len(set(task_order)) != expected_rows:
        raise CompatibilityError("queue task IDs missing/duplicate")
    verifier_index = _index(verifier, "verifier")
    solver_index = _index(v2_solver, "v2 solver")
    router_index = _index(router, "Router")
    if set(verifier_index) != set(task_order):
        raise CompatibilityError("verifier task set mismatch")
    if set(solver_index) != set(task_order):
        raise CompatibilityError("v2 solver task set mismatch")
    if set(router_index) != set(task_order):
        raise CompatibilityError("Router task set mismatch")

    output: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    decisions: collections.Counter[str] = collections.Counter()
    for index, queue_row in enumerate(queue):
        task_id = task_order[index]
        verifier_row = verifier_index[task_id]
        solver_row = solver_index[task_id]
        router_row = router_index[task_id]
        if verifier_row.get("queue_request_sha256") != queue_row.get("request_sha256"):
            raise CompatibilityError(f"{task_id}: verifier request binding mismatch")
        selection = verifier_row.get("selection")
        if not isinstance(selection, Mapping):
            raise CompatibilityError(f"{task_id}: v2 selection audit missing")
        source_verifier_sha = _row_sha(verifier_row)
        source_solver_sha = _row_sha(solver_row)
        source_router_sha = _row_sha(router_row)
        reasons: list[str] = []
        recovered: dict[str, Any] | None = None
        if verifier_row.get("error"):
            if solver_row != router_row:
                raise CompatibilityError(f"{task_id}: v2 error fallback is not exact Router")
            if (
                selection.get("selected_source") != "router"
                or selection.get("reason") != "verifier_error_router_fallback"
                or selection.get("router_row_sha256") != source_router_sha
                or selection.get("gold_access") is not False
                or str(selection.get("applied_final_answer") or "")
                != str(router_row.get("final_answer") or "")
            ):
                raise CompatibilityError(f"{task_id}: v2 error selection is not bound Router")
            recovered, reasons = assess_numeric_compatibility(
                verifier_row=verifier_row, queue_row=queue_row
            )
        if recovered is not None:
            decision = "numeric_choice_token_compat_recovered"
            composed = _compose_recovered_solver(
                verifier_row=verifier_row,
                verdict=recovered,
                queue_sha256=queue_sha256,
                compat_profile_sha256=compat_profile_sha256,
                source_verifier_sha256=source_verifier_sha,
                source_v2_solver_sha256=source_solver_sha,
            )
        elif verifier_row.get("error"):
            decision = "nonrecoverable_error_exact_router"
            composed = copy.deepcopy(dict(router_row))
        else:
            selected_source = selection.get("selected_source")
            if selection.get("router_row_sha256") != source_router_sha:
                raise CompatibilityError(f"{task_id}: v2 Router SHA audit mismatch")
            if selection.get("gold_access") is not False:
                raise CompatibilityError(f"{task_id}: v2 selection is not gold-blind")
            if selected_source == "meta_verifier":
                generation = solver_row.get("generation")
                if (
                    selection.get("reason") != "valid_supported_meta_answer"
                    or str(selection.get("applied_final_answer") or "")
                    != str(solver_row.get("final_answer") or "")
                    or solver_row.get("error") is not None
                    or not isinstance(generation, Mapping)
                    or generation.get("gold_access") is not False
                    or generation.get("queue_sha256") != queue_sha256
                    or generation.get("queue_request_sha256")
                    != queue_row.get("request_sha256")
                ):
                    raise CompatibilityError(f"{task_id}: invalid v2 meta-verifier row")
            elif selected_source == "router":
                allowed_reasons = {
                    "explicit_abstention_router_fallback",
                    "format_not_verified_router_fallback",
                    "confidence_gate_router_fallback",
                    "evidence_gate_router_fallback",
                }
                if (
                    selection.get("reason") not in allowed_reasons
                    or solver_row != router_row
                    or str(selection.get("applied_final_answer") or "")
                    != str(router_row.get("final_answer") or "")
                ):
                    raise CompatibilityError(f"{task_id}: invalid v2 Router row")
            else:
                raise CompatibilityError(f"{task_id}: unknown v2 selection source")
            decision = "unchanged_v2_content_exact"
            composed = copy.deepcopy(dict(solver_row))
        _assert_no_evaluation_keys(composed)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "queue_index": index,
            "task_id": task_id,
            "queue_request_sha256": queue_row.get("request_sha256"),
            "source_verifier_row_sha256": source_verifier_sha,
            "source_v2_solver_row_sha256": source_solver_sha,
            "source_router_row_sha256": source_router_sha,
            "decision": decision,
            "compatibility_rejection_reasons": reasons,
            "applied_final_answer": str(composed.get("final_answer") or ""),
            "output_row_sha256": _row_sha(composed),
            "raw_recovered_verdict": recovered,
            "gold_access": False,
            "task_id_or_subject_used_for_selection": False,
        }
        _assert_no_evaluation_keys(audit)
        decisions[decision] += 1
        output.append(composed)
        audits.append(audit)
    return output, audits, dict(sorted(decisions.items()))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(v2.preparation.stable_json(dict(row)) + "\n")


def _source(path: Path, rows: int | None = None) -> dict[str, Any]:
    value = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def validate_source_manifest(
    *,
    run_manifest: Mapping[str, Any],
    queue_path: Path,
    verifier_path: Path,
    solver_path: Path,
    router_path: Path,
    v2_profile_path: Path,
    preparation_manifest_path: Path,
) -> None:
    if run_manifest.get("complete") is not True:
        raise CompatibilityError("v2 run manifest is not complete")
    if run_manifest.get("generation_gold_access") is not False:
        raise CompatibilityError("v2 run manifest is not gold-blind")
    bindings = [
        ("queue", queue_path),
        ("verdict_output", verifier_path),
        ("solver_output", solver_path),
        ("router_fallback_solver", router_path),
        ("profile", v2_profile_path),
        ("preparation_manifest", preparation_manifest_path),
    ]
    for key, path in bindings:
        value = run_manifest.get(key)
        if not isinstance(value, Mapping) or value.get("sha256") != _sha256_file(path):
            raise CompatibilityError(f"v2 run manifest {key} SHA binding mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--v2-solver", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--v2-run-manifest", type=Path, required=True)
    parser.add_argument("--v2-profile", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--compat-profile", type=Path, default=DEFAULT_COMPAT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    _assert_file_sha(args.queue, EXPECTED_QUEUE_SHA256, "queue")
    _assert_file_sha(
        args.preparation_manifest,
        EXPECTED_PREPARATION_MANIFEST_SHA256,
        "v2 preparation manifest",
    )
    _assert_file_sha(args.v2_profile, EXPECTED_V2_PROFILE_SHA256, "v2 profile")
    _assert_file_sha(args.router, EXPECTED_ROUTER_SHA256, "Router")
    _assert_file_sha(args.compat_profile, EXPECTED_PROFILE_SHA256, "compat profile")
    compat_profile = _read_json(args.compat_profile, "compat profile")
    validate_compat_profile(compat_profile)
    v2_profile = _read_json(args.v2_profile, "v2 profile")
    v2.preparation.validate_profile(v2_profile)
    v2.validate_queue(_read_jsonl(args.queue, "queue"))
    source_manifest = _read_json(args.v2_run_manifest, "v2 run manifest")
    validate_source_manifest(
        run_manifest=source_manifest,
        queue_path=args.queue,
        verifier_path=args.verifier,
        solver_path=args.v2_solver,
        router_path=args.router,
        v2_profile_path=args.v2_profile,
        preparation_manifest_path=args.preparation_manifest,
    )
    queue = _read_jsonl(args.queue, "queue")
    verifier = _read_jsonl(args.verifier, "verifier")
    v2_solver = _read_jsonl(args.v2_solver, "v2 solver")
    router = _read_jsonl(args.router, "Router")
    solver, audit, decisions = compose_rows(
        queue=queue,
        verifier=verifier,
        v2_solver=v2_solver,
        router=router,
        queue_sha256=EXPECTED_QUEUE_SHA256,
        compat_profile_sha256=EXPECTED_PROFILE_SHA256,
    )

    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        solver_path = temporary / "solver.jsonl"
        audit_path = temporary / "compatibility_audit.jsonl"
        manifest_path = temporary / "composition_manifest.json"
        _write_jsonl(solver_path, solver)
        _write_jsonl(audit_path, audit)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "condition": CONDITION,
            "gold_access": False,
            "score_or_judge_inputs_loaded": False,
            "task_id_or_subject_used_for_selection": False,
            "profile": _source(args.compat_profile),
            "sources": {
                "queue": _source(args.queue, len(queue)),
                "verifier": _source(args.verifier, len(verifier)),
                "v2_solver": _source(args.v2_solver, len(v2_solver)),
                "router": _source(args.router, len(router)),
                "v2_run_manifest": _source(args.v2_run_manifest),
                "v2_profile": _source(args.v2_profile),
                "v2_preparation_manifest": _source(args.preparation_manifest),
            },
            "code": _source(Path(__file__).resolve()),
            "outputs": {
                "solver": _source(solver_path, len(solver)),
                "compatibility_audit": _source(audit_path, len(audit)),
            },
            "decision_counts": decisions,
            "recursive_gold_free_audit": "PASS",
        }
        _write_json(manifest_path, manifest)
        files = [solver_path, audit_path, manifest_path]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(f"{_sha256_file(path)}  {path.name}\n" for path in files),
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists() and temporary.parent.resolve() == output.parent.resolve():
            shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "rows": len(solver),
                "decision_counts": decisions,
                "solver_sha256": _sha256_file(output / "solver.jsonl"),
                "audit_sha256": _sha256_file(output / "compatibility_audit.jsonl"),
                "manifest_sha256": _sha256_file(output / "composition_manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compose_maxim_meta_v3_choice_token_compat_v31 as composer


def _verdict(answer: str = "7") -> dict[str, Any]:
    return {
        "question_reconstruction": "A complete independent reconstruction",
        "decisive_evidence": ["First decisive fact", "Second decisive fact"],
        "candidate_checks": [
            {
                "candidate_id": candidate_id,
                "status": "supported",
                "verification": f"Independent check for {candidate_id}",
            }
            for candidate_id in composer.v3.preparation.OPAQUE_IDS
        ],
        "independent_reasoning": "Independent reasoning from the original image",
        "final_answer": answer,
        "confidence": 0.95,
        "answer_format_verified": True,
        "abstain": False,
    }


def _router(task_id: str, answer: str = "8") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "condition": "maxim_subject_router_v1",
        "model": "Qwen/Qwen3.5-9B",
        "prompt_version": "v2_cot",
        "final_answer": answer,
        "reasoning": "frozen Router reasoning",
        "solution_steps": "frozen Router steps",
        "forced_answer": False,
        "generation": {"gold_access": False},
        "offline_provenance": {
            "gold_access": False,
            "benchmark_fields_used": ["task_id", "subject"],
        },
        "tool_calls": [],
        "usage": {},
        "error": None,
    }


def _error_verifier(task_id: str, queue_request: str, answer: str = "7") -> dict[str, Any]:
    router = _router(task_id)
    return {
        "task_id": task_id,
        "queue_request_sha256": queue_request,
        "verifier_request_sha256": "v" * 64,
        "prompt_version": "original-first-anonymous-12way-verification-v3",
        "model": "Qwen/Qwen3.5-27B",
        "verdict": None,
        "raw_response": json.dumps(_verdict(answer)),
        "call": {
            "finish_reason": "stop",
            "parse_error": None,
            "recovered_partial": False,
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_s": 2.0,
        },
        "semantic_attempt": 2,
        "error": f"{composer.STRICT_REJECTION} | {composer.STRICT_REJECTION}",
        "selection": {
            "selected_source": "router",
            "reason": "verifier_error_router_fallback",
            "applied_final_answer": router["final_answer"],
            "router_row_sha256": composer.audited_base._row_sha(router),
            "gold_access": False,
        },
    }


def test_generic_single_digit_recovery_passes_every_other_v3_gate() -> None:
    recovered, reasons = composer.assess_choice_token_compatibility(
        verifier_row=_error_verifier("synthetic_one", "q" * 64),
        queue_row={"task_id": "synthetic_one", "answer_type": "choice"},
    )
    assert reasons == []
    assert recovered is not None
    assert recovered["final_answer"] == "7"
    assert len(recovered["candidate_checks"]) == 12


def test_transport_length_repetition_and_mixed_errors_stay_nonrecoverable() -> None:
    queue = {"task_id": "synthetic_two", "answer_type": "choice"}
    for error in (
        "CallFailure: transport timeout | CallFailure: transport timeout",
        "CallFailure: invalid structured response finish_reason='length' | "
        "CallFailure: invalid structured response finish_reason='length'",
        "CallFailure: repeated completion | CallFailure: repeated completion",
        f"{composer.STRICT_REJECTION} | MetaVerifierError: evidence invalid",
    ):
        verifier = _error_verifier("synthetic_two", "a" * 64)
        verifier["error"] = error
        recovered, _ = composer.assess_choice_token_compatibility(
            verifier_row=verifier, queue_row=queue
        )
        assert recovered is None


def test_saved_call_raw_json_and_single_digit_are_strict() -> None:
    queue = {"task_id": "synthetic_three", "answer_type": "choice"}
    for mutation in ("partial", "parse", "length", "missing", "multi_digit"):
        verifier = _error_verifier("synthetic_three", "b" * 64)
        if mutation == "partial":
            verifier["call"]["recovered_partial"] = True
        elif mutation == "parse":
            verifier["call"]["parse_error"] = "invalid JSON"
        elif mutation == "length":
            verifier["call"]["finish_reason"] = "length"
        elif mutation == "missing":
            verifier["raw_response"] = None
        else:
            verifier["raw_response"] = json.dumps(_verdict("10"))
        recovered, _ = composer.assess_choice_token_compatibility(
            verifier_row=verifier, queue_row=queue
        )
        assert recovered is None


def test_policy_gates_and_nonchoice_are_not_recovered() -> None:
    queue = {"task_id": "synthetic_four", "answer_type": "choice"}
    for field, value in (
        ("confidence", 0.69),
        ("abstain", True),
        ("answer_format_verified", False),
    ):
        verifier = _error_verifier("synthetic_four", "c" * 64)
        raw = json.loads(verifier["raw_response"])
        raw[field] = value
        verifier["raw_response"] = json.dumps(raw)
        assert composer.assess_choice_token_compatibility(
            verifier_row=verifier, queue_row=queue
        )[0] is None
    assert composer.assess_choice_token_compatibility(
        verifier_row=_error_verifier("synthetic_four", "c" * 64),
        queue_row={"task_id": "synthetic_four", "answer_type": "numeric"},
    )[0] is None


def test_compose_recovers_only_eligible_row_and_copies_normal_v3_exact() -> None:
    request_one = "1" * 64
    request_two = "2" * 64
    queue = [
        {
            "task_id": "synthetic_one",
            "answer_type": "choice",
            "request_sha256": request_one,
        },
        {
            "task_id": "synthetic_two",
            "answer_type": "choice",
            "request_sha256": request_two,
        },
    ]
    router_one = _router("synthetic_one")
    router_two = _router("synthetic_two", "B")
    recovered_verifier = _error_verifier("synthetic_one", request_one)
    unchanged_solver = {
        "task_id": "synthetic_two",
        "condition": "maxim_final_gold_blind_meta_verifier_v3",
        "model": "Qwen/Qwen3.5-27B",
        "prompt_version": "original-first-anonymous-12way-verification-v3",
        "final_answer": "B",
        "reasoning": "already valid",
        "solution_steps": "two evidence rows",
        "forced_answer": False,
        "raw_response": "{}",
        "generation": {
            "gold_access": False,
            "queue_sha256": "f" * 64,
            "queue_request_sha256": request_two,
        },
        "tool_calls": [],
        "usage": {},
        "error": None,
    }
    unchanged_verifier = {
        "task_id": "synthetic_two",
        "queue_request_sha256": request_two,
        "verifier_request_sha256": "w" * 64,
        "error": None,
        "selection": {
            "selected_source": "meta_verifier",
            "reason": "valid_supported_meta_answer",
            "applied_final_answer": "B",
            "router_row_sha256": composer.audited_base._row_sha(router_two),
            "gold_access": False,
        },
    }
    output, audit, counts = composer.compose_rows(
        queue=queue,
        verifier=[recovered_verifier, unchanged_verifier],
        v3_solver=[router_one, unchanged_solver],
        router=[router_one, router_two],
        queue_sha256="f" * 64,
        compat_profile_sha256=composer.EXPECTED_PROFILE_SHA256,
        expected_rows=2,
    )
    assert output[0]["condition"] == composer.CONDITION
    assert output[0]["final_answer"] == "7"
    generation = output[0]["generation"]
    assert "source_v3_solver_row_sha256" in generation
    assert "source_v2_solver_row_sha256" not in generation
    assert generation["structured_mode"].endswith("v31")
    assert output[1] == unchanged_solver
    assert audit[0]["decision"] == "numeric_choice_token_compat_recovered"
    assert audit[1]["decision"] == "unchanged_v3_content_exact"
    assert counts == {
        "numeric_choice_token_compat_recovered": 1,
        "unchanged_v3_content_exact": 1,
    }


def test_profile_and_code_are_generic_and_gold_blind() -> None:
    profile = json.loads(composer.DEFAULT_COMPAT_PROFILE.read_text(encoding="utf-8"))
    composer.validate_compat_profile(profile)
    text = Path(composer.__file__).read_text(encoding="utf-8")
    assert "val_0177" not in text
    assert "subject.casefold" not in text
    assert "reference_answer" not in text
    assert "judge_score" not in text

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compose_maxim_meta_v2_choice_token_compat_v21 as composer


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
            for candidate_id in composer.v2.preparation.OPAQUE_IDS
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
        "prompt_version": "original-first-anonymous-10way-verification-v2",
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
            "router_row_sha256": composer._row_sha(router),
            "gold_access": False,
        },
    }


def test_generic_numeric_recovery_passes_every_other_frozen_gate() -> None:
    queue = {"task_id": "synthetic_one", "answer_type": "choice"}
    verifier = _error_verifier("synthetic_one", "q" * 64)
    recovered, reasons = composer.assess_numeric_compatibility(
        verifier_row=verifier, queue_row=queue
    )
    assert reasons == []
    assert recovered is not None
    assert recovered["final_answer"] == "7"
    assert recovered["confidence"] == 0.95


def test_mixed_or_missing_evidence_failures_are_not_recovered() -> None:
    queue = {"task_id": "synthetic_two", "answer_type": "choice"}
    mixed = _error_verifier("synthetic_two", "a" * 64)
    mixed["error"] = (
        f"{composer.STRICT_REJECTION} | MetaVerifierError: decisive_evidence must contain 2..8 rows"
    )
    assert composer.assess_numeric_compatibility(
        verifier_row=mixed, queue_row=queue
    )[0] is None

    no_raw = _error_verifier("synthetic_two", "a" * 64)
    no_raw["raw_response"] = None
    assert composer.assess_numeric_compatibility(
        verifier_row=no_raw, queue_row=queue
    )[0] is None

    too_long = _error_verifier("synthetic_two", "a" * 64, answer="10")
    assert composer.assess_numeric_compatibility(
        verifier_row=too_long, queue_row=queue
    )[0] is None


def test_policy_gates_and_nonchoice_are_not_recovered() -> None:
    queue = {"task_id": "synthetic_three", "answer_type": "choice"}
    low = _error_verifier("synthetic_three", "b" * 64)
    raw = json.loads(low["raw_response"])
    raw["confidence"] = 0.69
    low["raw_response"] = json.dumps(raw)
    recovered, reasons = composer.assess_numeric_compatibility(
        verifier_row=low, queue_row=queue
    )
    assert recovered is None
    assert "confidence_gate_failed" in reasons

    abstained = _error_verifier("synthetic_three", "b" * 64)
    raw = json.loads(abstained["raw_response"])
    raw["abstain"] = True
    abstained["raw_response"] = json.dumps(raw)
    assert composer.assess_numeric_compatibility(
        verifier_row=abstained, queue_row=queue
    )[0] is None

    assert composer.assess_numeric_compatibility(
        verifier_row=_error_verifier("synthetic_three", "b" * 64),
        queue_row={"task_id": "synthetic_three", "answer_type": "numeric"},
    )[0] is None


def test_all_rows_are_composed_with_exact_unchanged_and_router_semantics() -> None:
    request_one = "1" * 64
    request_two = "2" * 64
    queue = [
        {"task_id": "synthetic_one", "answer_type": "choice", "request_sha256": request_one},
        {"task_id": "synthetic_two", "answer_type": "choice", "request_sha256": request_two},
    ]
    router_one = _router("synthetic_one")
    router_two = _router("synthetic_two", "B")
    recovered_verifier = _error_verifier("synthetic_one", request_one)
    unchanged_solver = {
        "task_id": "synthetic_two",
        "condition": "maxim_final_gold_blind_meta_verifier_v2",
        "model": "Qwen/Qwen3.5-27B",
        "prompt_version": "original-first-anonymous-10way-verification-v2",
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
            "router_row_sha256": composer._row_sha(router_two),
            "gold_access": False,
        },
    }
    output, audit, counts = composer.compose_rows(
        queue=queue,
        verifier=[recovered_verifier, unchanged_verifier],
        v2_solver=[router_one, unchanged_solver],
        router=[router_one, router_two],
        queue_sha256="f" * 64,
        compat_profile_sha256=composer.EXPECTED_PROFILE_SHA256,
        expected_rows=2,
    )
    assert output[0]["condition"] == composer.CONDITION
    assert output[0]["final_answer"] == "7"
    assert output[0]["generation"]["source_rejection"] == composer.STRICT_REJECTION
    assert output[1] == unchanged_solver
    assert audit[0]["decision"] == "numeric_choice_token_compat_recovered"
    assert audit[1]["decision"] == "unchanged_v2_content_exact"
    assert counts == {
        "numeric_choice_token_compat_recovered": 1,
        "unchanged_v2_content_exact": 1,
    }


def test_composer_contains_no_task_or_subject_special_case() -> None:
    text = (SCRIPTS / "compose_maxim_meta_v2_choice_token_compat_v21.py").read_text(
        encoding="utf-8"
    )
    assert "val_0177" not in text
    assert "Biology" not in text
    assert "Math" not in text
    assert "subject.casefold" not in text

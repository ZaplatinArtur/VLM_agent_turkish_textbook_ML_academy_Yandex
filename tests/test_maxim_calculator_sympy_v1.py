from __future__ import annotations

import json
from pathlib import Path

import pytest

from mla_baseline.calculator_sympy import (
    answer_equivalent,
    decide_calculator_switch,
    execute_program,
    safe_math_equivalent,
    sympy_available,
)
from scripts import run_maxim_calculator_sympy_v1 as runner


def _draft(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "applicable": True,
        "problem_kind": "percent",
        "reasoning": "120 sayisinin yuzde 15'i",
        "solution_steps": "120 * 15 / 100 = 18",
        "verification_program": "result = 120 * 15 / 100",
        "predicted_program_value": "18",
        "independent_answer": "B",
        "option_mapping": "18 -> B",
        "unit_check": "not_applicable",
        "constraint_check": "pass",
        "confidence": "high",
    }
    value.update(updates)
    return value


def _audit(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "switch_recommended": True,
        "tool_consistent": True,
        "question_consistent": True,
        "unit_check": "not_applicable",
        "constraint_check": "pass",
        "reasoning": "Gorsel ve gercek arac sonucu B'yi destekliyor.",
        "solution_steps": "120 * 15 / 100 = 18; 18 -> B",
        "final_answer": "B",
        "confidence": "high",
    }
    value.update(updates)
    return value


def test_calculator_uses_exact_bounded_arithmetic() -> None:
    result = execute_program("result = 120 * 15 / 100")

    assert result.ok is True
    assert result.value == "18"
    assert result.nontrivial is True
    assert result.operation_count == 2
    assert result.numeric_literal_count == 3


def test_calculator_uses_sympy_ast_without_sympify() -> None:
    if not sympy_available():
        pytest.skip("SymPy not installed in this environment")

    result = execute_program("result = sqrt(8) / sqrt(2)")

    assert result.ok is True
    assert result.value == "2"
    assert result.engine == "sympy_ast"
    assert safe_math_equivalent("60\N{DEGREE SIGN}", "pi/3 radians") is True
    assert safe_math_equivalent("2*x=2", "x=1") is True
    assert safe_math_equivalent("sqrt(8)/sqrt(2)", "2") is True


def test_calculator_supports_all_prompted_integer_helpers() -> None:
    assert execute_program("gcd(12, 18) + lcm(2, 3)").value == "12"
    assert execute_program("comb(5, 2) + perm(3, 2)").value == "16"
    assert execute_program("17 // 5 + 17 % 5").value == "5"


def test_combinatorics_are_bounded_before_expensive_integer_work() -> None:
    huge_comb = execute_program("comb(1000000000, 500000000)")
    huge_perm = execute_program("perm(1000000000, 2)")

    assert huge_comb.ok is False
    assert "safe range" in str(huge_comb.error)
    assert huge_perm.ok is False
    assert "safe range" in str(huge_perm.error)


def test_nested_powers_are_bounded_before_huge_integer_materialization() -> None:
    result = execute_program("(((2 ** 12) ** 12) ** 12) ** 12")

    assert result.ok is False
    assert "too large" in str(result.error)


def test_calculator_rejects_code_large_power_and_constant_answer() -> None:
    arbitrary = execute_program("__import__('os').system('whoami')")
    large_power = execute_program("2 ** 100")
    constant = execute_program("result = 18")

    assert arbitrary.ok is False
    assert large_power.ok is False
    assert constant.ok is True
    assert constant.nontrivial is False


def test_switch_needs_a_second_independent_audit() -> None:
    draft = _draft()
    program = execute_program(str(draft["verification_program"]))

    before_audit = decide_calculator_switch(
        baseline_answer="C",
        answer_type="choice",
        draft=draft,
        program=program,
    )
    after_audit = decide_calculator_switch(
        baseline_answer="C",
        answer_type="choice",
        draft=draft,
        program=program,
        audit=_audit(),
    )

    assert before_audit.switch is False
    assert before_audit.audit_required is True
    assert before_audit.reasons == ("audit_required",)
    assert after_audit.switch is True
    assert after_audit.audit_required is False
    assert after_audit.reasons == ()


@pytest.mark.parametrize(
    ("draft_updates", "expected_reason"),
    [
        ({"applicable": False}, "not_applicable"),
        ({"confidence": "medium"}, "draft_not_high_confidence"),
        ({"predicted_program_value": "19"}, "predicted_value_mismatch"),
        ({"unit_check": "fail"}, "draft_unit_check_failed"),
        ({"constraint_check": "fail"}, "draft_constraint_check_failed"),
        ({"constraint_check": ""}, "draft_constraint_check_failed"),
        (
            {"verification_program": "result = 18"},
            "trivial_program",
        ),
    ],
)
def test_first_stage_failures_preserve_frozen_candidate(
    draft_updates: dict[str, object], expected_reason: str
) -> None:
    draft = _draft(**draft_updates)
    program = execute_program(str(draft["verification_program"]))

    decision = decide_calculator_switch(
        baseline_answer="C",
        answer_type="choice",
        draft=draft,
        program=program,
    )

    assert decision.switch is False
    assert decision.audit_required is False
    assert expected_reason in decision.reasons


def test_agreement_needs_no_comparison_call() -> None:
    draft = _draft(independent_answer="C", option_mapping="18 -> C")
    program = execute_program(str(draft["verification_program"]))

    decision = decide_calculator_switch(
        baseline_answer="C",
        answer_type="choice",
        draft=draft,
        program=program,
    )

    assert decision.switch is False
    assert decision.audit_required is False
    assert decision.reasons == ("independent_agrees_with_baseline",)


def test_numeric_answer_must_match_actual_program_value() -> None:
    draft = _draft(
        independent_answer="5",
        predicted_program_value="4",
        verification_program="result = 2 + 2",
    )
    program = execute_program(str(draft["verification_program"]))

    decision = decide_calculator_switch(
        baseline_answer="3",
        answer_type="numeric",
        draft=draft,
        program=program,
        audit=_audit(final_answer="5"),
    )

    assert program.value == "4"
    assert decision.switch is False
    assert decision.audit_required is False
    assert decision.reasons == ("independent_program_disagree",)


def test_numeric_equation_substitution_requires_zero_residual() -> None:
    draft = _draft(
        problem_kind="equation_substitution",
        independent_answer="5",
        predicted_program_value="0",
        verification_program="result = 2 * 5 - 10",
    )
    program = execute_program(str(draft["verification_program"]))

    before_audit = decide_calculator_switch(
        baseline_answer="3",
        answer_type="numeric",
        draft=draft,
        program=program,
    )
    bad_residual = decide_calculator_switch(
        baseline_answer="3",
        answer_type="numeric",
        draft={
            **draft,
            "predicted_program_value": "1",
            "verification_program": "result = 2 * 5 - 9",
        },
        program=execute_program("result = 2 * 5 - 9"),
    )

    assert before_audit.audit_required is True
    assert bad_residual.switch is False
    assert bad_residual.audit_required is False
    assert "substitution_residual_nonzero" in bad_residual.reasons


@pytest.mark.parametrize(
    ("audit_updates", "expected_reason"),
    [
        ({"switch_recommended": False}, "audit_rejected_switch"),
        ({"tool_consistent": False}, "audit_tool_inconsistent"),
        ({"question_consistent": False}, "audit_question_inconsistent"),
        ({"confidence": "medium"}, "audit_not_high_confidence"),
        ({"unit_check": "fail"}, "audit_unit_check_failed"),
        ({"constraint_check": "fail"}, "audit_constraint_check_failed"),
        ({"final_answer": "A"}, "independent_audit_disagree"),
    ],
)
def test_second_stage_conflicts_preserve_frozen_candidate(
    audit_updates: dict[str, object], expected_reason: str
) -> None:
    draft = _draft()
    program = execute_program(str(draft["verification_program"]))

    decision = decide_calculator_switch(
        baseline_answer="C",
        answer_type="choice",
        draft=draft,
        program=program,
        audit=_audit(**audit_updates),
    )

    assert decision.switch is False
    assert expected_reason in decision.reasons


def test_answer_equivalence_is_type_aware() -> None:
    assert answer_equivalent("B", "B) 18", "choice") is True
    assert answer_equivalent("1/2", "0,5", "numeric") is True
    assert answer_equivalent("x=1", "2*x=2", "free_form") is True
    assert answer_equivalent("A", "B", "choice") is False


def _task(image_name: str) -> dict[str, object]:
    return {
        "task_id": "task_safe_001",
        "subject": "Math",
        "grade": 6,
        "question": "Question visible in image",
        "question_images": [
            {
                "image_id": "question",
                "format": "file_path",
                "data": image_name,
                "mime_type": "image/png",
            }
        ],
        "answer_type": "choice",
        "reference_answer": "SECRET_GOLD",
        "reference_solution": "SECRET_SOLUTION",
    }


def _baseline() -> dict[str, object]:
    return {
        "task_id": "task_safe_001",
        "condition": "agent_rag",
        "model": "Qwen/Qwen3.5-9B",
        "prompt_version": "v2_cot",
        "final_answer": "C",
        "solution_steps": "frozen steps",
        "reasoning": "frozen reasoning",
        "forced_answer": False,
        "raw_response": "{}",
        "generation": {"call_count": 1, "gold_access": False},
        "tool_calls": [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "latency_s": 1.0},
        "error": None,
    }


def test_first_call_is_candidate_blind_and_all_calls_are_gold_blind(
    tmp_path: Path,
) -> None:
    image = tmp_path / "question.png"
    image.write_bytes(b"not-read-by-message-builder")
    redacted = runner.gold_blind_task(_task(image.name))
    first = runner._first_messages(
        redacted,
        image_root=tmp_path,
        image_url_root="http://127.0.0.1:18080",
    )
    program = execute_program("result = 120 * 15 / 100")
    audit = runner._audit_messages(
        redacted,
        {**_baseline(), "final_answer": "ZZZ_FROZEN_CANDIDATE"},
        _draft(),
        program,
        image_root=tmp_path,
        image_url_root="file:///images",
    )

    first_text = json.dumps(first, ensure_ascii=False)
    audit_text = json.dumps(audit, ensure_ascii=False)
    assert "SECRET_GOLD" not in first_text + audit_text
    assert "SECRET_SOLUTION" not in first_text + audit_text
    assert "reference_answer" not in first_text + audit_text
    assert "ZZZ_FROZEN_CANDIDATE" not in first_text
    assert "ZZZ_FROZEN_CANDIDATE" in audit_text
    assert "http://127.0.0.1:18080/question.png" in first_text


def test_full_benchmark_dry_run_does_not_create_output_or_expose_gold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "question.png"
    image.write_bytes(b"placeholder")
    tasks = tmp_path / "tasks.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    output = tmp_path / "must_not_exist.jsonl"
    tasks.write_text(json.dumps(_task(image.name)) + "\n", encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()) + "\n", encoding="utf-8")

    result = runner.main(
        [
            "--input",
            str(tasks),
            "--baseline-results",
            str(baseline),
            "--image-root",
            str(tmp_path),
            "--output",
            str(output),
            "--allow-unfrozen-sources",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr().out

    assert result == 0
    assert not output.exists()
    assert "SECRET_GOLD" not in captured
    assert '"gold_access": false' in captured
    assert '"math_tasks": 1' in captured


def test_runner_rejects_unfrozen_sources_by_default(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    tasks.write_text(json.dumps(_task("question.png")) + "\n", encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="frozen source SHA256 mismatch"):
        runner.main(
            [
                "--input",
                str(tasks),
                "--baseline-results",
                str(baseline),
                "--image-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--dry-run",
            ]
        )


def test_non_math_task_is_exact_passthrough_without_endpoint_call() -> None:
    class NoCalls:
        def complete(self, **_: object) -> dict[str, object]:
            raise AssertionError("endpoint must not be called for non-Math")

    task = runner.gold_blind_task(
        {
            **_task("question.png"),
            "subject": "History",
        }
    )
    baseline = _baseline()
    row = runner.run_task(
        task,
        baseline,
        pool=NoCalls(),  # type: ignore[arg-type]
        model="Qwen/Qwen3.5-9B",
        image_root=Path("."),
        image_url_root="file:///images",
        source_sha256={"tasks": "x", "baseline_results": "y"},
    )

    assert row["final_answer"] == baseline["final_answer"]
    assert row["reasoning"] == baseline["reasoning"]
    assert row["condition"] == runner.CONDITION
    trace = row["generation"]["calculator_sympy"]
    assert trace["eligibility"] == "non_math_passthrough"
    assert trace["call_traces"] == []
    assert trace["switch_applied"] is False
    assert row["usage"] == baseline["usage"]


def test_runner_applies_only_a_fully_verified_math_switch(tmp_path: Path) -> None:
    image = tmp_path / "question.png"
    image.write_bytes(b"placeholder")

    class FakePool:
        def __init__(self) -> None:
            self.responses = [_draft(), _audit()]

        def complete(self, **_: object) -> dict[str, object]:
            parsed = self.responses.pop(0)
            return {
                "parsed": parsed,
                "raw": json.dumps(parsed),
                "endpoint": "fake",
                "finish_reason": "stop",
                "attempt": 1,
                "latency_s": 0.01,
                "input_tokens": 20,
                "output_tokens": 10,
                "recovered_partial": False,
                "parse_error": None,
            }

    task = runner.gold_blind_task(_task(image.name))
    row = runner.run_task(
        task,
        _baseline(),
        pool=FakePool(),  # type: ignore[arg-type]
        model="Qwen/Qwen3.5-9B",
        image_root=tmp_path,
        image_url_root="file:///images",
        source_sha256={"tasks": "x", "baseline_results": "y"},
    )

    trace = row["generation"]["calculator_sympy"]
    assert row["final_answer"] == "B"
    assert trace["switch_applied"] is True
    assert len(trace["call_traces"]) == 2
    assert row["generation"]["call_count"] == 3
    assert row["tool_calls"][-1]["tool"] == "bounded_calculator_sympy"
    assert row["usage"]["input_tokens"] == 50
    assert row["usage"]["output_tokens"] == 25


def test_new_verifier_sources_have_no_task_overrides_or_hardcoded_benchmark_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "src" / "mla_baseline" / "calculator_sympy.py",
        root / "scripts" / "run_maxim_calculator_sympy_v1.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "TASK_OVERRIDES" not in source
    assert "REFERENCE_ANSWERS" not in source
    assert "GOLD_ANSWERS" not in source
    assert "val_0001" not in source

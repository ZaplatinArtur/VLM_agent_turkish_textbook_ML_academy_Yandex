from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_maxim_visual_sketchpad_v2 as sketchpad


def _plan(**overrides: object) -> dict:
    value = {
        "sketch_kind": "auxiliary_lines",
        "image_index": 0,
        "focus_bbox_1000": [400, 400, 600, 600],
        "points": [],
        "lines": [],
        "plan_notes": "Use the visible geometry.",
        "confidence": 0.95,
    }
    value.update(overrides)
    return value


def _solve(**overrides: object) -> dict:
    value = {
        "visual_facts": ["fact 1", "fact 2", "fact 3"],
        "sketch_specific_fact": "The marked anchors align.",
        "verification_checks": ["check 1", "check 2"],
        "sketch_helpful": True,
        "original_sketch_consistent": True,
        "answer_format_verified": True,
        "confidence": 0.95,
        "final_answer": "C",
        "reasoning": "safe reasoning",
        "solution_steps": "safe steps",
    }
    value.update(overrides)
    return value


def _task() -> dict:
    return {
        "task_id": "task-1",
        "subject": "Math",
        "grade": 7,
        "question": "Visible question",
        "question_images": [{"data": "page.png"}],
        "answer_type": "multiple_choice",
    }


def _fallback(answer: str = "A") -> dict:
    return {
        "task_id": "task-1",
        "condition": "frozen-active-crop-v2",
        "model": sketchpad.core.MODEL,
        "final_answer": answer,
        "solution_steps": "frozen steps",
        "reasoning": "frozen reasoning",
        "forced_answer": False,
        "raw_response": "frozen",
        "generation": {"gold_access": False, "frozen": True},
        "tool_calls": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "latency_s": 1.0},
        "error": None,
    }


def _decode_data_url(value: str) -> Image.Image:
    payload = base64.b64decode(value.split(",", 1)[1])
    return Image.open(io.BytesIO(payload)).convert("RGB")


def test_grid_is_drawn_only_for_coordinate_grid(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(page)
    auxiliary_url, _, _ = sketchpad._render_sketch(page, _plan())
    grid_url, _, _ = sketchpad._render_sketch(
        page, _plan(sketch_kind="coordinate_grid")
    )
    auxiliary = _decode_data_url(auxiliary_url)
    grid = _decode_data_url(grid_url)
    assert auxiliary.getpixel((10, 5)) == (255, 255, 255)
    assert grid.getpixel((10, 5)) != (255, 255, 255)


def test_choice_gate_requires_strict_letter_and_canonicalises_fallback() -> None:
    task = _task()
    same = sketchpad.gate_failures(task, _plan(), _solve(final_answer="B"), _fallback("b."))
    assert "same_answer_as_frozen_fallback" in same
    malformed = sketchpad.gate_failures(
        task, _plan(), _solve(final_answer="b."), _fallback("A")
    )
    assert "choice_candidate_not_strict_A_to_E" in malformed


class _FakePool:
    model = sketchpad.core.MODEL

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.messages: list[list[dict]] = []

    def complete(self, *, messages: list[dict], **_: object) -> dict:
        self.messages.append(messages)
        return {
            "parsed": self.results.pop(0),
            "endpoint": "fake",
            "finish_reason": "stop",
            "attempt": 1,
            "latency_s": 0.1,
            "input_tokens": 10,
            "output_tokens": 10,
            "recovered_partial": False,
            "parse_error": None,
        }


@pytest.mark.parametrize(
    ("kind", "confidence", "expected_failure"),
    [
        ("crop_box", 0.95, "ineligible_sketch_kind"),
        ("auxiliary_lines", 0.84, "planner_confidence_below_0.85"),
    ],
)
def test_planner_failures_short_circuit_second_call(
    tmp_path: Path, kind: str, confidence: float, expected_failure: str
) -> None:
    Image.new("RGB", (100, 100), "white").save(tmp_path / "page.png")
    pool = _FakePool([_plan(sketch_kind=kind, confidence=confidence)])
    row = sketchpad.run_task(
        _task(),
        _fallback(),
        pool=pool,
        image_root=tmp_path,
        image_url_root="file:///images",
    )
    assert len(pool.messages) == 1
    treatment = row["generation"]["visual_sketchpad_v2"]
    assert treatment["selected_source"] == "frozen_active_crop_v2"
    assert expected_failure in treatment["gate_failures"]
    assert treatment["sketch_metadata"]["render_skipped"] is True


def test_selected_image_index_is_clamped_before_render_and_solver_prompt(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (100, 100), "white").save(tmp_path / "page.png")
    pool = _FakePool([_plan(image_index=7), _solve(final_answer="C")])
    row = sketchpad.run_task(
        _task(),
        _fallback("A"),
        pool=pool,
        image_root=tmp_path,
        image_url_root="file:///images",
    )
    assert len(pool.messages) == 2
    treatment = row["generation"]["visual_sketchpad_v2"]
    assert treatment["selected_source"] == "visual_sketchpad_candidate"
    assert treatment["plan"]["image_index"] == 0
    assert treatment["sketch_metadata"]["requested_image_index"] == 7
    assert treatment["sketch_metadata"]["selected_image_index"] == 0
    assert '"image_index":0' in str(pool.messages[1])


def test_recursive_gold_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden keys"):
        sketchpad.rows_by_task(
            [{"task_id": "x", "nested": [{"gold_answer": "SECRET"}]}],
            label="test",
        )


def test_failclosed_preserves_frozen_answer_and_clears_top_level_error() -> None:
    fallback = _fallback("D")
    row = sketchpad._fallback_result(
        fallback,
        plan=None,
        sketch_metadata=None,
        solve=None,
        calls=[],
        failures=["candidate_error"],
        error="RuntimeError: test",
    )
    assert row["final_answer"] == "D"
    assert row["reasoning"] == fallback["reasoning"]
    assert row["condition"] == sketchpad.CONDITION
    assert row["error"] is None
    assert row["generation"]["gold_access"] is False


def test_open_answer_schema_allows_more_than_120_characters() -> None:
    assert sketchpad.SOLVE_SCHEMA["properties"]["final_answer"]["maxLength"] >= 1600

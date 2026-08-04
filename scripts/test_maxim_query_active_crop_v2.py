from __future__ import annotations

import base64
import copy
import sys
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compose_maxim_query_active_crop_v2 as compose
import prepare_maxim_query_active_crop_v2 as prepare
import run_maxim_query_active_crop_v2 as runner


def _geometry(pixels: int) -> list[dict[str, int]]:
    return [{"pixels": pixels}]


def test_blind_route_is_pinned_and_narrow_by_signal() -> None:
    assert prepare.route_reasons({"forced_answer": True}, _geometry(1)) == [
        "no_tools_forced_answer_parse_signal"
    ]
    assert prepare.route_reasons(
        {"forced_answer": False, "reasoning": "Grafik dikkatle okunmalı"},
        _geometry(600_000),
    ) == ["visual_anchor_plus_large_native_image"]
    assert prepare.route_reasons(
        {"forced_answer": False, "reasoning": "Sonuç muhtemelen A"},
        _geometry(600_000),
    ) == ["uncertainty_anchor_plus_large_native_image"]
    assert prepare.route_reasons(
        {"forced_answer": False, "reasoning": "Grafik"}, _geometry(599_999)
    ) == []


def test_public_payload_rejects_forbidden_fields_recursively() -> None:
    prepare.assert_public_payload({"task": {"answer_type": "choice"}})
    for key in ("reference_answer", "gold_answer", "judge_verdict", "score"):
        try:
            prepare.assert_public_payload({"nested": [{key: "SECRET"}]})
        except ValueError:
            pass
        else:
            raise AssertionError(f"forbidden key accepted: {key}")


def test_native_crop_is_valid_png_and_uses_exact_2_to_4x(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (1000, 1200), "white").save(source)
    url, metadata = runner._native_crop(source, [100, 100, 300, 300])
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["upscale_factor"] in {2, 3, 4}
    assert metadata["output_crop_size"][0] == (
        metadata["pixel_bbox"][2] - metadata["pixel_bbox"][0]
    ) * metadata["upscale_factor"]


def test_locator_and_solver_messages_contain_no_gold_secret(tmp_path: Path) -> None:
    Image.new("RGB", (300, 300), "white").save(tmp_path / "x.png")
    raw = {
        "task_id": "safe",
        "subject": "Math",
        "grade": 7,
        "question": "Visible",
        "question_images": [{"data": "images/x.png"}],
        "answer_type": "choice",
        "reference_answer": "SECRET-GOLD",
    }
    request = {
        "task": runner.core._task_view(raw),
        "fallback": {"final_answer": "A", "reasoning": "safe", "solution_steps": "safe"},
    }
    locate = runner._locator_messages(
        request, image_root=tmp_path, image_url_root="file:///images"
    )
    solve = runner._solver_messages(
        request,
        {"regions": [], "second_zoom_needed": False, "overall_confidence": 1.0},
        ["data:image/png;base64,AA=="],
        [{"safe": True}],
        image_root=tmp_path,
        image_url_root="file:///images",
    )
    assert "SECRET-GOLD" not in str(locate)
    assert "SECRET-GOLD" not in str(solve)


def _passing_result() -> dict:
    return {
        "task_id": "t1",
        "final_answer": "B",
        "error": None,
        "generation": {
            "gold_access": False,
            "locator": {
                "overall_confidence": 0.92,
                "used_regions": [{"confidence": 0.85}],
            },
            "selection_evidence": {
                "baseline_supported": False,
                "confidence": 0.94,
                "all_required_evidence_visible": True,
                "original_crop_consistent": True,
                "answer_format_verified": True,
                "visible_facts": ["fact 1", "fact 2"],
                "verification_checks": ["check 1", "check 2"],
            },
        },
    }


def test_selection_gate_passes_only_all_conjuncts() -> None:
    fallback = {"final_answer": "A"}
    passed, failures = compose.gate_decision(_passing_result(), fallback, "choice")
    assert passed and failures == []

    mutations = [
        ("error", "failure"),
        ("final_answer", "A"),
    ]
    for key, value in mutations:
        result = _passing_result()
        result[key] = value
        assert compose.gate_decision(result, fallback, "choice")[0] is False
    for key, value in (
        ("baseline_supported", True),
        ("confidence", 0.899),
        ("all_required_evidence_visible", False),
        ("original_crop_consistent", False),
        ("answer_format_verified", False),
        ("visible_facts", ["one"]),
        ("verification_checks", ["one"]),
    ):
        result = _passing_result()
        result["generation"]["selection_evidence"][key] = value
        assert compose.gate_decision(result, fallback, "choice")[0] is False


def test_compose_defaults_nonroute_and_failed_gate_to_no_tools() -> None:
    benchmark = [
        {"task_id": "t1", "answer_type": "choice"},
        {"task_id": "t2", "answer_type": "choice"},
    ]
    fallback = [
        {"task_id": "t1", "condition": "b0", "final_answer": "A"},
        {"task_id": "t2", "condition": "b0", "final_answer": "C"},
    ]
    queue = [{"task_id": "t1", "request_sha256": "request"}]
    passing = _passing_result()
    rows, details = compose.compose(
        benchmark_rows=benchmark,
        fallback_rows=fallback,
        queue_rows=queue,
        result_rows=[passing],
    )
    assert [row["final_answer"] for row in rows] == ["B", "C"]
    assert details["stats"]["active_crop_selected_rows"] == 1

    failed = copy.deepcopy(passing)
    failed["generation"]["selection_evidence"]["confidence"] = 0.4
    rows, details = compose.compose(
        benchmark_rows=benchmark,
        fallback_rows=fallback,
        queue_rows=queue,
        result_rows=[failed],
    )
    assert [row["final_answer"] for row in rows] == ["A", "C"]
    assert details["stats"]["active_crop_selected_rows"] == 0

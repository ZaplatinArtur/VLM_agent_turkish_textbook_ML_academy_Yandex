from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compose_maxim_staged_truncation_exact_web_overlay_v1 as composer


def exact_row(url: str, *, condition: str = composer.EXACT_WEB_CONDITION) -> dict:
    return {
        "task_id": "synthetic",
        "condition": condition,
        "model": "exact-official-web-key",
        "generation": {
            "exact_question_match": True,
            "explicit_official_answer_key": True,
        },
        "tool_calls": [{"url": url, "key_url": url}],
        "final_answer": "C",
    }


class OverlayPolicyTest(unittest.TestCase):
    def test_official_profile_accepts_only_allowlisted_hosts(self) -> None:
        policy = {
            "mode": "official_hosts_only",
            "allowed_hosts": ["dokuman.osym.gov.tr", "ogmmateryal.eba.gov.tr"],
        }
        self.assertTrue(
            composer.overlay_allowed(exact_row("https://dokuman.osym.gov.tr/key.pdf"), policy)
        )
        self.assertFalse(
            composer.overlay_allowed(exact_row("https://kurguluyorum.com/copy.pdf"), policy)
        )

    def test_exploratory_profile_includes_frozen_third_party_row(self) -> None:
        policy = {"mode": "all_frozen_exact_web_rows_including_third_party_copy"}
        self.assertTrue(
            composer.overlay_allowed(exact_row("https://kurguluyorum.com/copy.pdf"), policy)
        )

    def test_non_exact_row_never_overlays(self) -> None:
        policy = {"mode": "all_frozen_exact_web_rows_including_third_party_copy"}
        row = exact_row("https://dokuman.osym.gov.tr/key.pdf", condition="other")
        self.assertFalse(composer.overlay_allowed(row, policy))


class GoldIsolationTest(unittest.TestCase):
    def test_reference_field_is_rejected_recursively(self) -> None:
        with self.assertRaises(composer.CompositionError):
            composer.assert_no_reference_fields(
                {"task_id": "x", "nested": {"reference_answer": "C"}},
                "synthetic",
            )

    def test_public_shape_without_reference_is_accepted(self) -> None:
        composer.assert_no_reference_fields(
            {"task_id": "x", "subject": "Math", "answer_type": "choice"},
            "synthetic",
        )


class JudgeParityTest(unittest.TestCase):
    def test_solver_failure_overrides_positive_frozen_judge(self) -> None:
        judge = {"task_id": "x", "judge": {"error": None}, "verdict": {"strict_correct": True}}
        self.assertTrue(composer.strict_judge_value(judge))
        self.assertTrue(composer.solver_failed({"task_id": "x", "final_answer": ""}))


if __name__ == "__main__":
    unittest.main()

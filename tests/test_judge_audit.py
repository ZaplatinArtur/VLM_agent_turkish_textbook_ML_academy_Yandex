import unittest

from vlm_judge.judge_audit import audit_judge_run, validate_judge_completion


def _verdict():
    return {
        "label": "fully_correct",
        "score": 4,
        "strict_correct": True,
        "final_answer_correct": True,
        "reasoning_correct": True,
        "complete": True,
        "confidence": 0.9,
        "error_types": [],
        "rationale": "ok",
        "reference_quality_issue": False,
    }


class JudgeAuditTests(unittest.TestCase):
    def test_operational_summary(self) -> None:
        records = [
            {
                "request_id": "r1",
                "prompt_version": "judge-v1",
                "setup": "no_tools",
                "verdict": _verdict(),
                "judge": {
                    "model": "qwen",
                    "attempts": 1,
                    "cache_hit": False,
                    "error": None,
                    "response_metadata": {
                        "served_model": "qwen",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                },
            },
            {
                "request_id": "r2",
                "prompt_version": "judge-v1",
                "setup": "web_search",
                "verdict": None,
                "judge": {"model": "qwen", "attempts": 2, "cache_hit": False, "error": "timeout"},
            },
        ]
        report = audit_judge_run(records)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["schema_valid_rate_after_retries"], 0.5)
        self.assertEqual(report["token_totals"]["total"], 15)
        self.assertEqual(report["failed_records"], 1)

    def test_completion_requires_exact_error_free_valid_coverage(self) -> None:
        expected = [{"task_id": "q1"}, {"task_id": "q2"}]
        judge = [
            {"task_id": "q1", "verdict": _verdict(), "judge": {"error": None}},
            {"task_id": "q2", "verdict": _verdict(), "judge": {"error": None}},
        ]

        report = validate_judge_completion(expected, judge)

        self.assertTrue(report["valid"])
        self.assertEqual(report["expected_records"], 2)
        self.assertEqual(report["judge_records"], 2)

    def test_completion_reports_missing_duplicate_and_failed_records(self) -> None:
        expected = [{"task_id": "q1"}, {"task_id": "q2"}]
        judge = [
            {"task_id": "q1", "verdict": None, "judge": {"error": "timeout"}},
            {"task_id": "q1", "verdict": _verdict(), "judge": {"error": None}},
            {"task_id": "q3", "verdict": _verdict(), "judge": {"error": None}},
        ]

        report = validate_judge_completion(expected, judge)

        self.assertFalse(report["valid"])
        self.assertEqual(report["duplicate_judge_task_ids"], ["q1"])
        self.assertEqual(report["missing_task_ids"], ["q2"])
        self.assertEqual(report["unexpected_task_ids"], ["q3"])
        self.assertEqual(report["invalid_verdict_task_ids"], ["q1"])
        self.assertEqual(report["judge_error_task_ids"], ["q1"])


if __name__ == "__main__":
    unittest.main()

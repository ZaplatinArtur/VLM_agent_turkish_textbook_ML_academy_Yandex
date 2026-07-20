import unittest

from vlm_judge.judge_audit import audit_judge_run


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


if __name__ == "__main__":
    unittest.main()

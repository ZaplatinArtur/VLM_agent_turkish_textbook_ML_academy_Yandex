import unittest

from vlm_judge.aggregation import aggregate_results


class AggregationTests(unittest.TestCase):
    def test_paired_delta(self) -> None:
        records = [
            {
                "task_id": "1",
                "setup": "no_tools",
                "subject": "Math",
                "deterministic": {"applicable": True, "matched": False},
            },
            {
                "task_id": "1",
                "setup": "textbook_retrieval",
                "subject": "Math",
                "deterministic": {"applicable": True, "matched": True},
            },
            {
                "task_id": "2",
                "setup": "no_tools",
                "subject": "Math",
                "deterministic": {"applicable": True, "matched": True},
            },
            {
                "task_id": "2",
                "setup": "textbook_retrieval",
                "subject": "Math",
                "deterministic": {"applicable": True, "matched": True},
            },
        ]
        result = aggregate_results(records)
        comparison = result["paired_comparisons"]["textbook_retrieval_vs_no_tools"]
        self.assertEqual(comparison["paired_tasks"], 2)
        self.assertEqual(comparison["textbook_retrieval_wins"], 1)
        self.assertEqual(comparison["ties"], 1)
        self.assertEqual(result["by_setup"]["textbook_retrieval"]["strict_accuracy"], 1.0)

    def test_hybrid_prefers_exact_metric_and_counts_agent_failure(self) -> None:
        records = [
            {
                "task_id": "1",
                "setup": "no_tools",
                "subject": "Math",
                "answer_type": "multiple_choice",
                "deterministic": {"applicable": True, "matched": True},
                "verdict": {
                    "label": "incorrect", "strict_correct": False, "score": 0,
                    "final_answer_correct": False, "reasoning_correct": None,
                    "complete": True, "confidence": 0.9, "error_types": [],
                    "rationale": "wrong", "reference_quality_issue": False,
                },
            },
            {
                "task_id": "2",
                "setup": "no_tools",
                "subject": "Math",
                "answer_type": "open_ended",
                "metadata": {"agent_failure": True},
                "deterministic": {"applicable": False, "matched": None},
                "verdict": None,
            },
        ]
        result = aggregate_results(records)
        summary = result["by_setup"]["no_tools"]
        self.assertEqual(summary["strict_accuracy"], 0.5)
        self.assertEqual(summary["agent_failures_counted_incorrect"], 1)
        self.assertEqual(
            result["metric_views"]["judge_only"]["no_tools"]["strict_accuracy"],
            0.0,
        )

    def test_setup_specific_run_ids_do_not_break_task_pairing(self) -> None:
        records = [
            {
                "task_id": "q1",
                "setup": setup,
                "metadata": {"run_id": f"experiment-{setup}"},
                "deterministic": {"applicable": True, "matched": setup == "web_search"},
            }
            for setup in ("no_tools", "web_search")
        ]
        result = aggregate_results(records)
        comparison = result["paired_comparisons"]["web_search_vs_no_tools"]
        self.assertEqual(comparison["paired_tasks"], 1)
        self.assertEqual(comparison["web_search_wins"], 1)

    def test_text_binary_verdicts_are_aggregated(self) -> None:
        records = [
            {
                "task_id": "q1",
                "setup": "no_tools",
                "subject": "Math",
                "answer_type": "open_ended",
                "verdict": {"score": 1, "rationale": "matches"},
            },
            {
                "task_id": "q2",
                "setup": "no_tools",
                "subject": "Math",
                "answer_type": "open_ended",
                "verdict": {"score": 0, "rationale": "does not match"},
            },
        ]

        result = aggregate_results(records)
        summary = result["by_setup"]["no_tools"]

        self.assertEqual(summary["scored"], 2)
        self.assertEqual(summary["evaluation_failures"], 0)
        self.assertEqual(summary["score_sources"], {"judge_binary": 2})
        self.assertEqual(summary["strict_accuracy"], 0.5)
        self.assertEqual(summary["mean_score_0_4"], 2.0)

    def test_malformed_text_binary_verdict_is_an_evaluation_failure(self) -> None:
        records = [
            {
                "task_id": "q1",
                "setup": "no_tools",
                "verdict": {"score": 2, "rationale": "invalid binary score"},
            },
        ]

        result = aggregate_results(records)
        summary = result["by_setup"]["no_tools"]

        self.assertEqual(summary["scored"], 0)
        self.assertEqual(summary["evaluation_failures"], 1)


if __name__ == "__main__":
    unittest.main()

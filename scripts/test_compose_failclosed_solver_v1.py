from __future__ import annotations

import unittest

from scripts import compose_failclosed_solver_v1 as composer


def row(task_id: str, answer: str | None, *, error: str | None = None, condition: str = "x") -> dict:
    return {
        "task_id": task_id,
        "condition": condition,
        "prompt_version": condition,
        "final_answer": answer,
        "reasoning": "r",
        "solution_steps": "s",
        "model": "m",
        "generation": {"gold_access": False},
        "error": error,
    }


class FailclosedComposerTests(unittest.TestCase):
    def test_retains_valid_and_replaces_only_failure(self) -> None:
        benchmark = [{"task_id": "a", "reference_answer": "SECRET"}, {"task_id": "b"}]
        candidate = [row("a", "A"), row("b", None, error="timeout")]
        router = [row("a", "B", condition="router"), row("b", "C", condition="router")]
        output, stats = composer.compose(
            benchmark_rows=benchmark, candidate_rows=candidate, router_rows=router,
            condition="composite",
        )
        self.assertEqual([item["final_answer"] for item in output], ["A", "C"])
        self.assertEqual(stats["router_fallback_task_ids"], ["b"])
        self.assertEqual(output[0]["generation"]["failclosed_composition"]["chosen_source"], "candidate")
        self.assertEqual(output[1]["generation"]["failclosed_composition"]["chosen_source"], "frozen_subject_router")
        self.assertNotIn("SECRET", str(output))

    def test_rejects_task_set_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate task-id set"):
            composer.compose(
                benchmark_rows=[{"task_id": "a"}, {"task_id": "b"}],
                candidate_rows=[row("a", "A")],
                router_rows=[row("a", "A"), row("b", "B")],
                condition="composite",
            )

    def test_rejects_non_blind_source(self) -> None:
        candidate = row("a", "A")
        candidate["generation"]["gold_access"] = True
        with self.assertRaisesRegex(ValueError, "gold_access"):
            composer.compose(
                benchmark_rows=[{"task_id": "a"}], candidate_rows=[candidate],
                router_rows=[row("a", "B")], condition="composite",
            )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.dryrun import run_synthetic_experiment


class SyntheticDryRunTests(unittest.TestCase):
    def test_complete_experiment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.jsonl"
            tasks = [
                {
                    "task_id": "q1", "subject": "Math", "grade": 5,
                    "answer_type": "multiple_choice", "question_text": "2+2?",
                    "question_image_url": None, "reference_answer": "B",
                    "reference_image_url": None, "acceptable_answers": [], "metadata": {},
                },
                {
                    "task_id": "q2", "subject": "Math", "grade": 5,
                    "answer_type": "numeric", "question_text": "3+4?",
                    "question_image_url": None, "reference_answer": "7",
                    "reference_image_url": None, "acceptable_answers": [], "metadata": {},
                },
            ]
            benchmark.write_text(
                "".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8"
            )
            result = run_synthetic_experiment(benchmark, root / "dryrun")
            self.assertTrue(result["synthetic_smoke_test"])
            self.assertTrue(result["validation_ready"])
            self.assertEqual(result["candidate_records"], 6)
            report = root / "dryrun" / "report.html"
            self.assertIn("Синтетический dry run", report.read_text(encoding="utf-8"))
            summary = json.loads((root / "dryrun" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["paired_comparisons"]["web_search_vs_no_tools"]["paired_tasks"],
                2,
            )


if __name__ == "__main__":
    unittest.main()

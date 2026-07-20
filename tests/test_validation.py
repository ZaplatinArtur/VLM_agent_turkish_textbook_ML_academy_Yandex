import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.validation import validate_experiment_runs


class RunValidationTests(unittest.TestCase):
    def test_complete_grid_is_ready_despite_metadata_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "task_id": "q1", "subject": "Math", "grade": 5,
                "answer_type": "multiple_choice", "question_text": "2+2?",
                "question_image_url": None, "reference_answer": "B",
                "reference_image_url": None, "acceptable_answers": [], "metadata": {},
            }
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(json.dumps(task) + "\n", encoding="utf-8")
            paths = {}
            for setup in ("no_tools", "web_search", "textbook_retrieval"):
                path = root / f"{setup}.jsonl"
                path.write_text(
                    json.dumps({**task, "setup": setup, "candidate_answer": "B"}) + "\n",
                    encoding="utf-8",
                )
                paths[setup] = path
            report = validate_experiment_runs(benchmark, paths)
            self.assertTrue(report["ready_for_experiment"])
            self.assertEqual(report["grid"]["complete_grid_rate"], 1.0)
            self.assertGreater(report["warning_count"], 0)
            strict_report = validate_experiment_runs(benchmark, paths, strict_metadata=True)
            self.assertFalse(strict_report["ready_for_experiment"])
            self.assertGreater(strict_report["error_count"], 0)

    def test_missing_task_blocks_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "task_id": "q1", "subject": "Math", "grade": 5,
                "answer_type": "multiple_choice", "question_text": "2+2?",
                "question_image_url": None, "reference_answer": "B",
                "reference_image_url": None, "acceptable_answers": [], "metadata": {},
            }
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(json.dumps(task) + "\n", encoding="utf-8")
            report = validate_experiment_runs(benchmark, {})
            self.assertFalse(report["ready_for_experiment"])
            self.assertEqual(report["error_count"], 3)


if __name__ == "__main__":
    unittest.main()

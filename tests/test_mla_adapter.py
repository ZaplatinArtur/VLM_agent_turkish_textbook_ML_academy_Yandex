import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vlm_judge.mla_adapter import (
    build_seed_text_tasks,
    candidate_text_from_solve_result,
    prepare_text_judge_input,
)


class MlaAdapterTests(unittest.TestCase):
    def test_candidate_preserves_solution_and_final_answer(self):
        value = candidate_text_from_solve_result(
            {"solution_steps": "2+2=4", "final_answer": "4"}
        )
        self.assertIn("Solution:\n2+2=4", value)
        self.assertIn("Final answer:\n4", value)

    def test_join_maps_condition_to_setup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            results = root / "results.jsonl"
            output = root / "judge.jsonl"
            tasks.write_text(
                json.dumps({
                    "task_id": "q1", "subject": "math", "grade": 7,
                    "question": "2+2?", "reference_answer": "4",
                    "answer_type": "numeric",
                }) + "\n",
                encoding="utf-8",
            )
            results.write_text(
                json.dumps({
                    "task_id": "q1", "condition": "b0_no_tools",
                    "model": "qwen", "prompt_version": "v1",
                    "solution_steps": "2+2=4", "final_answer": "4",
                    "tool_calls": [], "usage": {}, "error": None,
                }) + "\n",
                encoding="utf-8",
            )
            report = prepare_text_judge_input(tasks, results, output, require_all=True)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["written"], 1)
            self.assertEqual(record["setup"], "no_tools")
            self.assertEqual(record["candidate_answer"], "Solution:\n2+2=4\n\nFinal answer:\n4")

    def test_extracts_real_seed_question(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "tasks.jsonl"
            source.write_text(
                json.dumps({
                    "task_id": "legacy_1",
                    "subject": "Math",
                    "answer_type": "multiple_choice",
                    "reference_answer": "D",
                    "candidate_answer": "### Question\nGerçek soru?\n\n### Solution\nYanlış çözüm",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report = build_seed_text_tasks(source, output)
            task = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["written"], 1)
            self.assertEqual(task["question"], "Gerçek soru?")
            self.assertEqual(task["answer_type"], "choice")


if __name__ == "__main__":
    unittest.main()

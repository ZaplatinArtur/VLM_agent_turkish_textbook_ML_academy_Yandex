import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vlm_judge.mla_adapter import (
    build_seed_text_tasks,
    candidate_text_from_solve_result,
    prepare_image_judge_input,
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

    def test_image_judge_input_keeps_question_and_reference_as_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "question.png").write_bytes(b"question")
            (images / "answer.png").write_bytes(b"answer")
            manifest = root / "manifest.jsonl"
            results = root / "results.jsonl"
            output = root / "judge.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "subject": "math",
                        "grade": 7,
                        "question_image": "images/question.png",
                        "reference_answer": None,
                        "reference_answer_image": "images/answer.png",
                        "answer_type": "choice",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            results.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "condition": "agent_rag",
                        "solution_steps": "work",
                        "final_answer": "B",
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = prepare_image_judge_input(
                manifest,
                results,
                root,
                output,
                require_all=True,
            )
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(report["reference_kinds"], {"text": 0, "image": 1})
            self.assertIsNone(record["question_text"])
            self.assertTrue(Path(record["question_image_url"]).is_absolute())
            self.assertTrue(Path(record["reference_image_url"]).is_absolute())
            self.assertEqual(record["answer_type"], "multiple_choice")
            self.assertEqual(record["setup"], "textbook_retrieval")


if __name__ == "__main__":
    unittest.main()

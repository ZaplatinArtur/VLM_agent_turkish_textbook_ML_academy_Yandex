import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.gold import apply_verified_gold
from vlm_judge.prompts import build_judge_request
from vlm_judge.schema import EvaluationItem


class GoldApplicationTests(unittest.TestCase):
    def test_verified_gold_is_applied_and_prompted_without_provenance_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "setup": "no_tools",
                        "subject": "math",
                        "answer_type": "multi_answer",
                        "question_text": "solve a and b",
                        "reference_answer": "old",
                        "acceptable_answers": ["legacy"],
                        "candidate_answer": "a=1, b=2",
                        "metadata": {"run_id": "secret-run"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gold = root / "gold.jsonl"
            gold.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "status": "verified",
                        "quality": "clear",
                        "transcription": "a=1; b=2",
                        "acceptable_answers": ["1, 2"],
                        "subanswers": ["a=1", "b=2"],
                        "notes": "Both parts are required.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "enriched.jsonl"
            report = apply_verified_gold(dataset, gold, output, require_all=True)
            enriched = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["applied_verified_gold"], 1)
            self.assertEqual(enriched["reference_answer"], "a=1; b=2")
            self.assertEqual(enriched["acceptable_answers"], ["legacy", "1, 2"])
            self.assertEqual(enriched["metadata"]["required_subanswers"], ["a=1", "b=2"])

            request = build_judge_request(EvaluationItem.from_dict(enriched))
            self.assertIn("required_subanswers", request.user_prompt)
            self.assertIn("Both parts are required.", request.user_prompt)
            self.assertNotIn("secret-run", request.user_prompt)
            self.assertNotIn("source_reference_answer_before_gold", request.user_prompt)


if __name__ == "__main__":
    unittest.main()

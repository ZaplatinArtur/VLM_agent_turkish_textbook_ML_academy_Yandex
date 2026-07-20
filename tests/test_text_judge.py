import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vlm_judge.backends import ReplayBackend
from vlm_judge.text_judge import (
    build_text_binary_request,
    evaluate_text_records,
    parse_text_binary_verdict,
)


class TextBinaryJudgeTests(unittest.TestCase):
    def test_request_is_text_only(self):
        request = build_text_binary_request("2+2?", "4", "4")
        self.assertEqual(request.image_urls, ())
        self.assertIn("QUESTION:", request.user_prompt)
        self.assertIn("REFERENCE:", request.user_prompt)
        self.assertIn("CANDIDATE:", request.user_prompt)

    def test_strict_binary_response(self):
        self.assertEqual(
            parse_text_binary_verdict('{"score": 1, "rationale": "matches"}')["score"],
            1,
        )
        with self.assertRaises(ValueError):
            parse_text_binary_verdict('{"score": 2, "rationale": "bad"}')
        with self.assertRaises(ValueError):
            parse_text_binary_verdict('{"score": 1, "rationale": "ok", "extra": true}')

    def test_end_to_end_replay(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            report = evaluate_text_records(
                [{
                    "task_id": "q1",
                    "question_text": "2+2?",
                    "reference_answer": "4",
                    "candidate_answer": "4",
                    "manual_score": 1,
                    "setup": "no_tools",
                }],
                ReplayBackend(['{"score": 1, "rationale": "matches"}']),
                output,
                retry_delay_seconds=0,
            )
            self.assertEqual(report["succeeded"], 1)
            saved = output.read_text(encoding="utf-8")
            self.assertIn('"agreement": true', saved)
            self.assertIn('"setup": "no_tools"', saved)


if __name__ == "__main__":
    unittest.main()

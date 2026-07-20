import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.arena import build_pairwise_records
from vlm_judge.ingest import import_candidates, read_records


class IngestAndArenaTests(unittest.TestCase):
    def test_csv_import_preserves_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "subject": "Math",
                        "answer_type": "multiple_choice",
                        "question_image_url": "https://yadi.sk/i/q",
                        "reference_answer": "A",
                    }
                ) + "\n",
                encoding="utf-8",
            )
            responses = root / "responses.csv"
            responses.write_text("task_id,answer,latency_ms\nq1,,1200\n", encoding="utf-8")
            output = root / "candidates.jsonl"
            summary = import_candidates(
                benchmark,
                responses,
                output,
                setup="no_tools",
            )
            record = read_records(output)[0]
            self.assertEqual(summary["empty_responses_preserved"], 1)
            self.assertEqual(record["candidate_answer"], "[EMPTY_RESPONSE]")
            self.assertTrue(record["metadata"]["agent_failure"])

    def test_mirrored_pairs_swap_sides(self) -> None:
        base = {
            "task_id": "q1",
            "subject": "Math",
            "candidate_answer": "",
            "metadata": {},
        }
        records = [
            {**base, "setup": "no_tools", "candidate_answer": "A"},
            {**base, "setup": "web_search", "candidate_answer": "B"},
        ]
        pairs = build_pairwise_records(
            records,
            "no_tools",
            "web_search",
            mirrored=True,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["candidate_a"], pairs[1]["candidate_b"])
        self.assertEqual(pairs[0]["candidate_b"], pairs[1]["candidate_a"])


if __name__ == "__main__":
    unittest.main()

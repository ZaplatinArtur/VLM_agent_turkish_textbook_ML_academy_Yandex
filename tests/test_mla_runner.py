import json
import tempfile
import unittest
from pathlib import Path

from mla_baseline.runner import load_done_ids


class MlaRunnerResumeTests(unittest.TestCase):
    def test_retry_errors_rewrites_only_successful_latest_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            records = [
                {"task_id": "ok", "error": None, "final_answer": "A"},
                {"task_id": "retry", "error": "timeout", "final_answer": None},
            ]
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )

            done = load_done_ids(output, retry_errors=True)
            remaining = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(done, {"ok"})
            self.assertEqual([record["task_id"] for record in remaining], ["ok"])


if __name__ == "__main__":
    unittest.main()

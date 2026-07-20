import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.annotation_dataset import (
    prepare_pointwise_annotation_dataset,
    sample_calibration_responses,
)


class AnnotationDatasetTests(unittest.TestCase):
    def test_combines_runs_with_setup_scoped_ids_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for setup in ("no_tools", "web_search"):
                path = root / f"{setup}.jsonl"
                path.write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "setup": setup,
                                "candidate_answer": setup,
                            }
                        )
                        for task_id in ("q1", "q2")
                    )
                    + "\n",
                    encoding="utf-8",
                )
                paths.append(path)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            report = prepare_pointwise_annotation_dataset(paths, first, seed="fixed")
            prepare_pointwise_annotation_dataset(paths, second, seed="fixed")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report["records"], 4)
            records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len({record["annotation_id"] for record in records}), 4)

            sample = root / "sample.jsonl"
            sample_report = sample_calibration_responses(paths, sample, size=2, seed="fixed")
            self.assertEqual(sample_report["selected_records"], 2)
            self.assertEqual(sample_report["selected_by_setup"], {"no_tools": 1, "web_search": 1})


if __name__ == "__main__":
    unittest.main()

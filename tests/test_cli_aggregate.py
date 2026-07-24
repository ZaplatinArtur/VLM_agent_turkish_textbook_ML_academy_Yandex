import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.cli import _load_aggregate_records


class AggregateCliTests(unittest.TestCase):
    def test_overlay_replaces_matching_task_and_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            overlay = root / "overlay.jsonl"
            base.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "q1",
                                "setup": "no_tools",
                                "verdict": {"score": 0, "rationale": "old"},
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "q2",
                                "setup": "no_tools",
                                "verdict": {"score": 1, "rationale": "unchanged"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            overlay.write_text(
                json.dumps(
                    {
                        "task_id": "q1",
                        "setup": "no_tools",
                        "verdict": {"score": 1, "rationale": "corrected"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records, replacements = _load_aggregate_records([base], [overlay])

        self.assertEqual(replacements, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["verdict"]["score"], 1)
        self.assertEqual(records[0]["verdict"]["rationale"], "corrected")

    def test_overlay_rejects_unknown_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            overlay = root / "overlay.jsonl"
            base.write_text(
                json.dumps({"task_id": "q1", "setup": "no_tools"}) + "\n",
                encoding="utf-8",
            )
            overlay.write_text(
                json.dumps({"task_id": "q2", "setup": "no_tools"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "has no base unit"):
                _load_aggregate_records([base], [overlay])


if __name__ == "__main__":
    unittest.main()

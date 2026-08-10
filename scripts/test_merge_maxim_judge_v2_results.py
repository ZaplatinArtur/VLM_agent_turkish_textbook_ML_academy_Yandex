from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import merge_maxim_judge_v2_results as merger  # noqa: E402


def judge_row(task_id: str, correct: bool, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": task_id,
        "prompt_version": "judge-v2",
        "verdict": {"strict_correct": correct},
        "judge": {"error": None},
    }
    row.update(updates)
    return row


class MergeRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.template = [{"task_id": task_id} for task_id in ("t3", "t1", "t2")]

    def test_merges_in_template_order_and_preserves_rows(self) -> None:
        reusable = [judge_row("t2", False)]
        fresh = [judge_row("t1", True), judge_row("t3", True)]

        rows, summary = merger.merge_rows(
            template_rows=self.template,
            reusable_rows=reusable,
            fresh_rows=fresh,
        )

        self.assertEqual([row["task_id"] for row in rows], ["t3", "t1", "t2"])
        self.assertIs(rows[2], reusable[0])
        self.assertEqual(
            summary["strict_correct"], {"reusable": 0, "fresh": 2, "total": 2}
        )
        self.assertEqual(summary["partition"]["missing_rows"], 0)

    def test_rejects_duplicate_within_a_partition(self) -> None:
        with self.assertRaisesRegex(merger.MergeError, "duplicate task_id t1"):
            merger.merge_rows(
                template_rows=self.template,
                reusable_rows=[judge_row("t1", True), judge_row("t1", False)],
                fresh_rows=[judge_row("t2", True), judge_row("t3", True)],
            )

    def test_rejects_overlap_between_partitions(self) -> None:
        with self.assertRaisesRegex(merger.MergeError, "overlap.*t1"):
            merger.merge_rows(
                template_rows=self.template,
                reusable_rows=[judge_row("t1", True), judge_row("t2", True)],
                fresh_rows=[judge_row("t1", True), judge_row("t3", True)],
            )

    def test_rejects_unexpected_task(self) -> None:
        with self.assertRaisesRegex(merger.MergeError, "unexpected.*other"):
            merger.merge_rows(
                template_rows=self.template,
                reusable_rows=[judge_row("t1", True)],
                fresh_rows=[
                    judge_row("t2", True),
                    judge_row("t3", True),
                    judge_row("other", True),
                ],
            )

    def test_rejects_missing_task_instead_of_substituting_a_verdict(self) -> None:
        with self.assertRaisesRegex(merger.MergeError, "incomplete.*t3"):
            merger.merge_rows(
                template_rows=self.template,
                reusable_rows=[judge_row("t1", True)],
                fresh_rows=[judge_row("t2", False)],
            )

    def test_rejects_duplicate_template_task(self) -> None:
        with self.assertRaisesRegex(merger.MergeError, "duplicate task_id t1"):
            merger.merge_rows(
                template_rows=[{"task_id": "t1"}, {"task_id": "t1"}],
                reusable_rows=[judge_row("t1", True)],
                fresh_rows=[],
            )

    def test_rejects_invalid_judge_schema_or_lineage(self) -> None:
        invalid_rows = (
            judge_row("t1", True, verdict={}),
            judge_row("t1", True, verdict={"strict_correct": 1}),
            judge_row("t1", True, verdict=None),
            judge_row("t1", True, prompt_version="judge-v1"),
            judge_row("t1", True, error="transport failure"),
            judge_row("t1", True, judge={"error": "parse failure"}),
        )
        for invalid in invalid_rows:
            with self.subTest(invalid=invalid):
                with self.assertRaises(merger.MergeError):
                    merger.merge_rows(
                        template_rows=[{"task_id": "t1"}],
                        reusable_rows=[invalid],
                        fresh_rows=[],
                    )


class BuildArtifactsTest(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_writes_hash_verified_manifest_and_checksum_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.jsonl"
            reusable = root / "reusable.jsonl"
            fresh = root / "fresh.jsonl"
            output = root / "merged.jsonl"
            manifest_path = root / "manifest.json"
            checksum_path = root / "merged.sha256"
            self.write_jsonl(template, [{"task_id": "b"}, {"task_id": "a"}])
            self.write_jsonl(reusable, [judge_row("a", True)])
            self.write_jsonl(fresh, [judge_row("b", False)])

            report = merger.build_from_paths(
                template_path=template,
                reusable_path=reusable,
                fresh_path=fresh,
                out_jsonl=output,
                out_manifest=manifest_path,
                out_sha256=checksum_path,
            )

            rows = merger.read_jsonl(output, "output")
            self.assertEqual([row["task_id"] for row in rows], ["b", "a"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], merger.SCHEMA_VERSION)
            self.assertEqual(manifest["output"]["sha256"], merger.sha256_file(output))
            self.assertEqual(manifest["validation"]["partition"]["fresh_rows"], 1)
            self.assertIn("no solver or routing inputs", manifest["operation"])
            self.assertEqual(report["manifest_sha256"], merger.sha256_file(manifest_path))
            checksum = checksum_path.read_text(encoding="utf-8")
            self.assertIn(merger.sha256_file(output), checksum)
            self.assertIn(merger.sha256_file(manifest_path), checksum)

    def test_refuses_to_overwrite_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.jsonl"
            reusable = root / "reusable.jsonl"
            fresh = root / "fresh.jsonl"
            output = root / "merged.jsonl"
            manifest = root / "manifest.json"
            checksum = root / "merged.sha256"
            self.write_jsonl(template, [{"task_id": "a"}])
            self.write_jsonl(reusable, [judge_row("a", True)])
            self.write_jsonl(fresh, [])
            output.write_text("user data", encoding="utf-8")

            with self.assertRaisesRegex(merger.MergeError, "refusing to overwrite"):
                merger.build_from_paths(
                    template_path=template,
                    reusable_path=reusable,
                    fresh_path=fresh,
                    out_jsonl=output,
                    out_manifest=manifest,
                    out_sha256=checksum,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "user data")
            self.assertFalse(manifest.exists())
            self.assertFalse(checksum.exists())


if __name__ == "__main__":
    unittest.main()

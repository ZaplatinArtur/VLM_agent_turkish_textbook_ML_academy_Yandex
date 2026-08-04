from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import build_maxim_summary as summary_builder


class MaximSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = summary_builder.build_summary()
        self.rows = {row["id"]: row for row in self.summary["rows"]}

    def test_expected_full_benchmark_scores(self) -> None:
        expected = {
            "page_rag": (141, 62, 79),
            "no_tools_frozen": (191, 105, 86),
            "direct_rf_v2": (137, 65, 72),
            "element_proxy": (123, 56, 67),
            "decompose_rf_v2": (150, 75, 75),
            "parallel8_rf_v2": (151, 79, 72),
            "graph253": (129, 57, 72),
        }
        for key, values in expected.items():
            with self.subTest(key=key):
                row = self.rows[key]
                self.assertEqual(values, (
                    row["overall"]["correct"],
                    row["math"]["correct"],
                    row["non_math"]["correct"],
                ))
                self.assertEqual(274, row["overall"]["n"])
                self.assertEqual(139, row["math"]["n"])
                self.assertEqual(135, row["non_math"]["n"])

    def test_subject_counts_reconstruct_totals(self) -> None:
        self.assertEqual(274, sum(row["n"] for row in self.summary["subjects"]))
        for key in self.rows:
            reconstructed = sum(row["correct"][key] for row in self.summary["subjects"])
            self.assertEqual(self.rows[key]["overall"]["correct"], reconstructed)

    def test_matched_comparisons_and_status_caveats(self) -> None:
        matched = self.summary["matched_comparisons"]
        self.assertAlmostEqual(0.09837064844049181, matched["decompose_vs_direct"]["mcnemar_exact_two_sided_p"])
        self.assertAlmostEqual(0.006610751152038574, matched["parallel8_vs_direct"]["math"]["mcnemar_exact_two_sided_p"])
        self.assertEqual("proxy_only_not_colpali", self.summary["method_status"]["element_proxy"])
        self.assertEqual("safe_partial_solutions_disabled", self.summary["method_status"]["graph253"])
        self.assertFalse(self.summary["headline"]["general_upgrade_found"])

    def test_hashes_match_files(self) -> None:
        for relative, expected in self.summary["artifact_sha256"].items():
            with self.subTest(path=relative):
                self.assertEqual(expected, summary_builder._sha256(summary_builder.REPO / relative))

    def test_checked_in_reports_match_builder(self) -> None:
        report_dir = summary_builder.REPORT_DIR
        expected_json = json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        expected_md = summary_builder.render_markdown(self.summary)
        self.assertEqual(expected_json, (report_dir / "SUMMARY.json").read_text(encoding="utf-8"))
        self.assertEqual(expected_md, (report_dir / "SUMMARY.md").read_text(encoding="utf-8"))
        self.assertIn("Non-Math", expected_md)
        self.assertIn("не ColPali", expected_md)


if __name__ == "__main__":
    unittest.main()

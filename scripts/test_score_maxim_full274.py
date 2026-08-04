from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_maxim_full274 import (
    ScoringError,
    build_report,
    sha256_file,
    write_reports,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Full274ScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.benchmark_path = root / "benchmark.jsonl"
        self.solver_path = root / "solver.jsonl"
        self.image_judge_path = root / "image_judge.jsonl"
        self.baseline_path = root / "baseline.jsonl"
        self.out_json = root / "score.json"
        self.out_md = root / "score.md"
        self.out_sha = root / "score.sha256"

        self.benchmark = [
            {
                "task_id": "t1",
                "subject": "Math",
                "answer_type": "choice",
                "reference_answer": "B",
            },
            {
                "task_id": "t2",
                "subject": "Science",
                "answer_type": "numeric",
                "reference_answer": "1,5",
            },
            {
                "task_id": "t3",
                "subject": "Math",
                "answer_type": "short_text",
                "reference_answer": "hello-world",
            },
            {
                "task_id": "t4",
                "subject": "Math",
                "answer_type": "free_form",
                "reference_answer": "http://reference-image.invalid/t4.png",
            },
        ]
        self.solver = [
            {
                "task_id": task_id,
                "condition": "synthetic",
                "model": "unit-test",
                "final_answer": answer,
                "generation": {"gold_access": False, "call_count": 2},
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "latency_s": latency,
                },
                "error": None,
            }
            for task_id, answer, latency in (
                ("t1", "B", 1.0),
                ("t2", "1.5000004 kg", 2.0),
                ("t3", "HELLO, world!", 3.0),
                ("t4", "drawn result", 4.0),
            )
        ]
        self.baseline = [
            {
                "task_id": task_id,
                "metadata": {"score_source": source},
                "verdict": {"strict_correct": correct},
                "judge": {"error": None},
            }
            for task_id, source, correct in (
                ("t1", "exact", False),
                ("t2", "exact", True),
                ("t3", "exact", True),
                ("t4", "vlm_image_judge", False),
            )
        ]
        self.image_judge = [
            {
                "task_id": "t4",
                "verdict": {"strict_correct": True},
                "judge": {"error": None},
            }
        ]
        self._flush()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _flush(self) -> None:
        _write_jsonl(self.benchmark_path, self.benchmark)
        _write_jsonl(self.solver_path, self.solver)
        _write_jsonl(self.image_judge_path, self.image_judge)
        _write_jsonl(self.baseline_path, self.baseline)

    def _build(self) -> dict:
        return build_report(
            benchmark_path=self.benchmark_path,
            solver_results_path=self.solver_path,
            image_judge_path=self.image_judge_path,
            baseline_judge_path=self.baseline_path,
            expected_rows=4,
            expected_deterministic=3,
            expected_image_judge=1,
            expected_benchmark_sha256=None,
            expected_baseline_judge_sha256=None,
        )

    def test_scores_fixed_denominator_sources_flips_and_usage(self) -> None:
        report = self._build()

        self.assertEqual(report["overall"]["n"], 4)
        self.assertEqual(report["overall"]["new_correct"], 4)
        self.assertEqual(report["overall"]["baseline_correct"], 2)
        self.assertEqual(report["by_source"]["deterministic"]["n"], 3)
        self.assertEqual(report["by_source"]["image_judge"]["n"], 1)
        self.assertEqual(
            report["changes_vs_frozen_page_rag"]["fixed_task_ids"], ["t1", "t4"]
        )
        self.assertEqual(report["changes_vs_frozen_page_rag"]["regressed_task_ids"], [])
        self.assertEqual(report["by_subject"]["Math"]["new_correct"], 3)
        self.assertEqual(report["operational"]["tokens"]["combined_tokens_total"], 44)
        self.assertEqual(report["operational"]["latency"]["latency_s_median"], 2.5)
        self.assertEqual(report["operational"]["model_calls"]["call_count_total"], 8)

        hashes = write_reports(
            report,
            out_json=self.out_json,
            out_md=self.out_md,
            out_sha256=self.out_sha,
        )
        self.assertEqual(hashes["json"], sha256_file(self.out_json))
        self.assertIn(hashes["json"], self.out_sha.read_text(encoding="utf-8"))
        rendered_json = json.loads(self.out_json.read_text(encoding="utf-8"))
        serialized = json.dumps(rendered_json)
        self.assertNotIn('"reference_answer":', serialized)
        self.assertNotIn('"final_answer":', serialized)

    def test_generation_failure_overrides_true_image_judge(self) -> None:
        self.solver[-1]["final_answer"] = None
        self.solver[-1]["error"] = "endpoint timeout"
        # A complete hybrid judge JSONL is accepted; only the frozen image
        # partition is consumed for the new score.
        self.image_judge = [
            {
                "task_id": row["task_id"],
                "verdict": {
                    "strict_correct": True if row["task_id"] == "t4" else False
                },
            }
            for row in self.benchmark
        ]
        self._flush()

        report = self._build()

        self.assertEqual(report["overall"]["new_correct"], 3)
        self.assertEqual(report["overall"]["n"], 4)
        self.assertEqual(
            report["operational"]["errors"][
                "judge_true_overridden_by_generation_failure_task_ids"
            ],
            ["t4"],
        )
        self.assertEqual(report["guardrails"]["image_judge_input_shape"], "full_hybrid")

    def test_rejects_duplicate_or_missing_solver_rows(self) -> None:
        self.solver.append(dict(self.solver[0]))
        self._flush()
        with self.assertRaisesRegex(ScoringError, "duplicate task_id t1"):
            self._build()

        self.solver = self.solver[:-2]
        self._flush()
        with self.assertRaisesRegex(ScoringError, "task-ID mismatch"):
            self._build()

    def test_rejects_incomplete_image_judge(self) -> None:
        self.image_judge = [
            {"task_id": "t1", "verdict": {"strict_correct": False}}
        ]
        self._flush()
        with self.assertRaisesRegex(ScoringError, "IDs must be exactly"):
            self._build()

    def test_rejects_solver_gold_fields_and_nonfalse_gold_access(self) -> None:
        self.solver[0]["reference_answer"] = "B"
        self._flush()
        with self.assertRaisesRegex(ScoringError, "forbidden gold fields"):
            self._build()

        del self.solver[0]["reference_answer"]
        self.solver[0]["generation"]["gold_access"] = True
        self._flush()
        with self.assertRaisesRegex(ScoringError, "gold_access=True"):
            self._build()


if __name__ == "__main__":
    unittest.main()

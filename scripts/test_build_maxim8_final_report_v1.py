from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_maxim8_final_report_v1 as builder


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Maxim8FinalReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.benchmark = self.root / "benchmark.jsonl"
        with self.benchmark.open("w", encoding="utf-8", newline="\n") as sink:
            for index in range(274):
                subject = "Math" if index < 139 else "Biology"
                sink.write(json.dumps({"task_id": f"val_{index:04d}", "subject": subject}) + "\n")
        self.scorer = self.root / "score_maxim.py"
        self.scorer.write_text("# frozen scorer\n", encoding="utf-8")
        self.benchmark_sha = _sha(self.benchmark)
        self.scorer_sha = _sha(self.scorer)

        baseline_score = self._score(correct=141, math_correct=62, fixed=0, regressed=0)
        exact_score = self._score(correct=165, math_correct=82, fixed=30, regressed=6)
        replay_score = self._score(correct=184, math_correct=105, fixed=54, regressed=11)
        _write_json(self.root / "baseline.json", baseline_score)
        _write_json(self.root / "idea_exact.json", exact_score)
        _write_json(self.root / "idea_replay.json", replay_score)
        _write_json(
            self.root / "idea_replay_manifest.json",
            {
                "sources": {"benchmark": {"sha256": self.benchmark_sha}},
                "image_judge_v2": {
                    "current_score_bounds_before_fresh_judge": {
                        "lower_correct": 163,
                        "upper_correct": 196,
                        "n": 274,
                        "lower_accuracy": round(163 / 274, 6),
                        "upper_accuracy": round(196 / 274, 6),
                    }
                },
            },
        )
        _write_json(
            self.root / "idea_pending_manifest.json",
            {
                "sources": {"benchmark": {"sha256": self.benchmark_sha}},
                "score_bounds": {
                    "lower_correct": 100,
                    "upper_correct": 182,
                    "n": 274,
                },
            },
        )
        _write_json(
            self.root / "ops.json",
            {"run": {"calls": 417, "tokens": 123456}},
        )
        exact = {
            "id": "idea_1",
            "name": "Exact idea",
            "matched_score": {
                "path": "idea_exact.json",
                "judge_lineage": "judge-v2",
                "matched": True,
            },
            "operational_manifest": {
                "path": "ops.json",
                "model_calls_pointer": "/run/calls",
                "combined_tokens_pointer": "/run/tokens",
            },
        }
        replay = {
            "id": "idea_2",
            "name": "Replay idea",
            "matched_score": {
                "path": "future_matched_score.json",
                "judge_lineage": "judge-v2",
                "matched": True,
            },
            "replay_score": {
                "path": "idea_replay.json",
                "judge_lineage": "mixed-historical",
                "matched": False,
            },
            "manifest": {"path": "idea_replay_manifest.json"},
        }
        pending = {
            "id": "idea_3",
            "name": "Pending idea",
            "matched_score": {
                "path": "missing.json",
                "judge_lineage": "judge-v2",
                "matched": True,
            },
            "manifest": {"path": "idea_pending_manifest.json"},
        }
        remaining = [
            {
                "id": f"idea_{index}",
                "name": f"Exact idea {index}",
                "matched_score": {
                    "path": "idea_exact.json",
                    "judge_lineage": "judge-v2",
                    "matched": True,
                },
            }
            for index in range(4, 9)
        ]
        self.config = {
            "schema_version": builder.CONFIG_SCHEMA_VERSION,
            "benchmark": {
                "path": "benchmark.jsonl",
                "sha256": self.benchmark_sha,
                "n": 274,
                "math_n": 139,
                "non_math_n": 135,
            },
            "scorer": {"path": "score_maxim.py", "sha256": self.scorer_sha},
            "matched_judge_lineage": "judge-v2",
            "baseline_reference": {"correct": 141, "n": 274},
            "baseline": {
                "id": "basic_page_rag",
                "name": "Basic page RAG",
                "matched_score": {
                    "path": "baseline.json",
                    "judge_lineage": "judge-v2",
                    "matched": True,
                },
            },
            "ideas": [exact, replay, pending, *remaining],
        }
        self.config_path = self.root / "config.json"
        self._save_config()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _save_config(self) -> None:
        _write_json(self.config_path, self.config)

    def _score(
        self, *, correct: int, math_correct: int, fixed: int, regressed: int
    ) -> dict[str, object]:
        non_math_correct = correct - math_correct
        return {
            "overall": {
                "n": 274,
                "new_correct": correct,
                "new_accuracy": round(correct / 274, 6),
                "baseline_correct": 141,
                "delta_correct": correct - 141,
                "fixed": fixed,
                "regressed": regressed,
            },
            "by_subject": {
                "Math": {
                    "n": 139,
                    "new_correct": math_correct,
                    "new_accuracy": round(math_correct / 139, 6),
                },
                "Biology": {
                    "n": 135,
                    "new_correct": non_math_correct,
                    "new_accuracy": round(non_math_correct / 135, 6),
                },
            },
            "operational": {
                "model_calls": {"call_count_total": 274, "reported_rows": 274},
                "tokens": {"combined_tokens_total": 9999},
            },
            "provenance": {
                "benchmark": {"sha256": self.benchmark_sha},
                "scorer": {"sha256": self.scorer_sha},
            },
        }

    def _configure_judge_evidence(self) -> tuple[Path, Path]:
        profile = {
            "lineage": "judge-v2",
            "prompt_version": "judge-v2",
            "model": "judge-model",
            "seed": 7,
            "temperature": 0.0,
            "max_tokens": 900,
            "enable_thinking": False,
            "use_response_format": True,
            "image_mode": "data_url",
            "backend": "openai-compatible",
        }
        rows = [
            {
                "task_id": f"image_{index}",
                "prompt_version": "judge-v2",
                "verdict": {"strict_correct": index == 0},
                "judge": {
                    "backend": "openai-compatible",
                    "backend_config_hash": "config-hash",
                    "error": None,
                    "model": "judge-model",
                    "backend_config": {
                        "model": "judge-model",
                        "seed": 7,
                        "temperature": 0.0,
                        "max_tokens": 900,
                        "enable_thinking": False,
                        "use_response_format": True,
                        "image_mode": "data_url",
                    },
                },
            }
            for index in range(2)
        ]
        artifact_path = self.root / "judge.jsonl"
        _write_jsonl(artifact_path, rows)
        artifact_sha = _sha(artifact_path)
        manifest_path = self.root / "judge_manifest.json"
        _write_json(
            manifest_path,
            {
                "matched_judge_lineage": "judge-v2",
                "matched_judge_profile": profile,
                "output": {
                    "path": "judge.jsonl",
                    "sha256": artifact_sha,
                },
            },
        )
        score_path = self.root / "idea_exact.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["provenance"]["image_judge"] = {"sha256": artifact_sha}
        _write_json(score_path, score)
        self.config["matched_judge_profile"] = profile
        self.config["ideas"][0]["matched_score"].update(
            {
                "judge_manifest": {"path": "judge_manifest.json"},
                "judge_artifact": {"path": "judge.jsonl"},
            }
        )
        self._save_config()
        return artifact_path, manifest_path

    def test_builds_baseline_and_exactly_eight_ideas(self) -> None:
        report = builder.build_report(self.config_path)
        self.assertEqual("exact", report["baseline"]["status"])
        self.assertEqual(8, len(report["ideas"]))
        self.assertEqual(
            {"exact": 6, "replay": 1, "partial": 0, "pending": 1},
            report["status_counts"],
        )
        exact, replay, pending = report["ideas"][:3]
        self.assertEqual((165, 82, 83), (
            exact["overall"]["correct"],
            exact["math"]["correct"],
            exact["non_math"]["correct"],
        ))
        self.assertEqual({"delta_correct": 24, "fixed": 30, "regressed": 6}, exact["vs_baseline"])
        self.assertEqual({"model_calls": 417, "combined_tokens": 123456}, exact["operational"])
        self.assertEqual("replay", replay["status"])
        self.assertFalse(replay["judge"]["matched"])
        self.assertEqual(184, replay["overall"]["correct"])
        self.assertIsNone(replay["matched_bounds"])
        self.assertIsNone(replay["pre_judge_bounds"])
        self.assertEqual("pending", pending["status"])
        self.assertIsNone(pending["overall"])
        self.assertEqual(182, pending["matched_bounds"]["overall"]["upper_correct"])
        self.assertEqual(pending["matched_bounds"], pending["pre_judge_bounds"])

    def test_markdown_labels_replay_and_pending_without_promoting_them(self) -> None:
        markdown = builder.render_markdown(builder.build_report(self.config_path))
        self.assertIn("184/274 (67.153%) (replay)", markdown)
        self.assertIn("mixed-historical (not matched)", markdown)
        self.assertIn(
            "| Pending idea | pending | — | pending | pending | pending |", markdown
        )
        self.assertNotIn("163–196/274", markdown)
        self.assertIn("100–182/274", markdown)
        self.assertIn("Pre-judge bounds", markdown)
        self.assertIn("`partial` reports measured incomplete progress only", markdown)

    def test_explicit_measured_progress_is_partial_but_never_accuracy(self) -> None:
        manifest_path = self.root / "idea_pending_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["progress"] = {
            "completed_rows": 91,
            "total_rows": 274,
            "unit": "tasks",
            "stage": "solver generation",
            "measured_at_utc": "2026-08-03T08:00:00Z",
        }
        _write_json(manifest_path, manifest)
        report = builder.build_report(self.config_path)
        partial = report["ideas"][2]
        self.assertEqual("partial", partial["status"])
        self.assertIsNone(partial["overall"])
        self.assertNotIn("accuracy", partial["progress"])
        self.assertEqual(
            (91, 274, "tasks"),
            (
                partial["progress"]["completed"],
                partial["progress"]["total"],
                partial["progress"]["unit"],
            ),
        )
        self.assertEqual(182, partial["pre_judge_bounds"]["overall"]["upper_correct"])
        markdown = builder.render_markdown(report)
        self.assertIn("91/274 tasks; solver generation", markdown)
        self.assertIn("partial; no full-bench score", markdown)

    def test_bounds_alone_do_not_create_partial_and_invalid_progress_is_rejected(self) -> None:
        report = builder.build_report(self.config_path)
        self.assertEqual("pending", report["ideas"][2]["status"])
        self.config["ideas"][2]["progress"] = {
            "completed": 12,
            "total": 97,
            "unit": "judge rows",
        }
        self._save_config()
        report = builder.build_report(self.config_path)
        self.assertEqual("partial", report["ideas"][2]["status"])
        self.assertEqual(12, report["ideas"][2]["progress"]["completed"])
        self.config["ideas"][2]["progress"] = {
            "completed": 275,
            "total": 274,
            "unit": "tasks",
        }
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "exceeds total"):
            builder.build_report(self.config_path)

    def test_exact_score_validates_configured_judge_manifest_and_artifact(self) -> None:
        self._configure_judge_evidence()
        exact = builder.build_report(self.config_path)["ideas"][0]
        self.assertEqual("exact", exact["status"])
        self.assertTrue(exact["judge"]["evidence"]["validated"])
        self.assertEqual("artifact+manifest", exact["judge"]["evidence"]["method"])
        self.assertEqual(2, exact["judge"]["evidence"]["artifact"]["rows"])

    def test_hybrid_artifact_validates_deterministic_and_profiled_rows(self) -> None:
        profile = {
            "lineage": "judge-v2",
            "prompt_version": "judge-v2",
            "model": "judge-model",
            "seed": 7,
        }
        deterministic_row = {
            "task_id": "deterministic",
            "prompt_version": "historical-hybrid",
            "deterministic": {"applicable": True, "matched": True},
            "metadata": {"score_source": "exact"},
            "verdict": {"strict_correct": True},
            "judge": {"backend": "hybrid-import", "error": None},
        }
        judge_row = {
            "task_id": "profiled",
            "prompt_version": "judge-v2",
            "deterministic": {"applicable": False, "matched": None},
            "verdict": {"strict_correct": False},
            "judge": {
                "backend": "openai-compatible",
                "model": "judge-model",
                "backend_config": {"model": "judge-model", "seed": 7},
                "error": None,
            },
        }
        artifact = self.root / "hybrid_judge.jsonl"
        _write_jsonl(artifact, [deterministic_row, judge_row])
        details = builder._validate_judge_artifact(
            artifact,
            expected_profile=profile,
            where="hybrid",
        )
        self.assertEqual(2, details["rows"])
        self.assertEqual(1, details["deterministic_rows"])
        self.assertEqual(1, details["profiled_judge_rows"])

        _write_jsonl(artifact, [deterministic_row])
        with self.assertRaisesRegex(builder.ReportConfigError, "no rows from the matched judge"):
            builder._validate_judge_artifact(
                artifact,
                expected_profile=profile,
                where="hybrid",
            )

        bypass = copy.deepcopy(deterministic_row)
        bypass["metadata"] = {"score_source": "untrusted"}
        _write_jsonl(artifact, [bypass, judge_row])
        with self.assertRaisesRegex(builder.ReportConfigError, "unrecognized.*score_source"):
            builder._validate_judge_artifact(
                artifact,
                expected_profile=profile,
                where="hybrid",
            )

        mismatch = copy.deepcopy(deterministic_row)
        mismatch["deterministic"]["matched"] = False
        _write_jsonl(artifact, [mismatch, judge_row])
        with self.assertRaisesRegex(builder.ReportConfigError, "must equal"):
            builder._validate_judge_artifact(
                artifact,
                expected_profile=profile,
                where="hybrid",
            )

    def test_manifest_can_resolve_and_validate_its_judge_artifact(self) -> None:
        self._configure_judge_evidence()
        del self.config["ideas"][0]["matched_score"]["judge_artifact"]
        self._save_config()
        evidence = builder.build_report(self.config_path)["ideas"][0]["judge"]["evidence"]
        self.assertTrue(evidence["validated"])
        self.assertEqual("artifact+manifest", evidence["method"])

    def test_configured_judge_evidence_rejects_wrong_actual_profile(self) -> None:
        artifact_path, manifest_path = self._configure_judge_evidence()
        rows = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines()]
        rows[1]["judge"]["backend_config"]["seed"] = 8
        _write_jsonl(artifact_path, rows)
        artifact_sha = _sha(artifact_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output"]["sha256"] = artifact_sha
        _write_json(manifest_path, manifest)
        score_path = self.root / "idea_exact.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["provenance"]["image_judge"]["sha256"] = artifact_sha
        _write_json(score_path, score)
        with self.assertRaisesRegex(builder.ReportConfigError, "seed=.*does not match"):
            builder.build_report(self.config_path)

    def test_configured_judge_evidence_requires_profile_and_score_hash_link(self) -> None:
        artifact_path, _ = self._configure_judge_evidence()
        del self.config["matched_judge_profile"]
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "profile is missing"):
            builder.build_report(self.config_path)
        self.config["matched_judge_profile"] = {
            "lineage": "judge-v2",
            "prompt_version": "judge-v2",
            "model": "judge-model",
            "seed": 7,
        }
        score_path = self.root / "idea_exact.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["provenance"]["image_judge"]["sha256"] = "0" * 64
        _write_json(score_path, score)
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "does not match configured artifact"):
            builder.build_report(self.config_path)

    def test_rejects_duplicate_ids_and_wrong_idea_count(self) -> None:
        self.config["ideas"][1]["id"] = "idea_1"
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "duplicate"):
            builder.build_report(self.config_path)
        self.config["ideas"] = self.config["ideas"][:-1]
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "exactly 8"):
            builder.build_report(self.config_path)

    def test_rejects_non_274_benchmark_and_hash_mismatch(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["benchmark"]["n"] = 273
        _write_json(self.config_path, bad)
        with self.assertRaisesRegex(builder.ReportConfigError, "274"):
            builder.build_report(self.config_path)
        bad = copy.deepcopy(self.config)
        bad["scorer"]["sha256"] = "0" * 64
        _write_json(self.config_path, bad)
        with self.assertRaisesRegex(builder.ReportConfigError, "scorer SHA256 mismatch"):
            builder.build_report(self.config_path)

    def test_requires_frozen_141_baseline_reference(self) -> None:
        self.config["baseline_reference"]["correct"] = 140
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "141/274"):
            builder.build_report(self.config_path)

    def test_rejects_score_provenance_and_accuracy_inconsistency(self) -> None:
        score_path = self.root / "idea_exact.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["provenance"]["scorer"]["sha256"] = "f" * 64
        _write_json(score_path, score)
        with self.assertRaisesRegex(builder.ReportConfigError, "scorer provenance"):
            builder.build_report(self.config_path)
        score["provenance"]["scorer"]["sha256"] = self.scorer_sha
        score["overall"]["new_accuracy"] = 0.1
        _write_json(score_path, score)
        with self.assertRaisesRegex(builder.ReportConfigError, "inconsistent"):
            builder.build_report(self.config_path)

    def test_rejects_replay_marked_as_matched_or_wrong_exact_lineage(self) -> None:
        self.config["ideas"][1]["replay_score"]["matched"] = True
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "matched must be false"):
            builder.build_report(self.config_path)
        self.config["ideas"][1]["replay_score"]["matched"] = False
        self.config["ideas"][0]["matched_score"]["judge_lineage"] = "judge-v1"
        self._save_config()
        with self.assertRaisesRegex(builder.ReportConfigError, "does not match"):
            builder.build_report(self.config_path)

    def test_cli_writer_refuses_overwrite_by_default(self) -> None:
        report = builder.build_report(self.config_path)
        output_json = self.root / "out" / "SUMMARY.json"
        output_md = self.root / "out" / "SUMMARY.md"
        builder._write_outputs(report, output_json, output_md, overwrite=False)
        self.assertEqual(report, json.loads(output_json.read_text(encoding="utf-8")))
        self.assertIn("# Maksim", output_md.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            builder._write_outputs(report, output_json, output_md, overwrite=False)


if __name__ == "__main__":
    unittest.main()

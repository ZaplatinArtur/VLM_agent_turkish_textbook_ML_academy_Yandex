from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import finalize_maxim_online_evaluation_v1 as finalizer  # noqa: E402
import prepare_maxim_online_evaluation_v1 as preparation  # noqa: E402
import score_maxim_full274 as scorer  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def solver_row(
    task_id: str, answer: str, steps: str, *, online: bool
) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": task_id,
        "condition": "test",
        "final_answer": answer,
        "solution_steps": steps,
        "error": None,
    }
    if online:
        row["generation"] = {"gold_access": False, "call_count": 1}
    return row


def judge_row(
    task_id: str, source: str, correct: bool, prompt: str
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "prompt_version": prompt,
        "metadata": {"score_source": source},
        "verdict": {"strict_correct": correct},
        "judge": {"error": None},
    }


def bound_fresh_judge_row(
    queue_row: dict[str, object], correct: bool
) -> dict[str, object]:
    request_id = finalizer.expected_fresh_request_id(queue_row)
    config = dict(finalizer.FROZEN_JUDGE_BACKEND_CONFIG)
    config_hash = finalizer._canonical_sha256(config)
    cache_key = hashlib.sha256(
        f"{request_id}:{config_hash}".encode("ascii")
    ).hexdigest()
    return {
        "request_id": request_id,
        "prompt_version": "judge-v2",
        "task_id": queue_row["task_id"],
        "setup": queue_row["setup"],
        "subject": queue_row["subject"],
        "grade": queue_row["grade"],
        "answer_type": queue_row["answer_type"],
        "metadata": queue_row["metadata"],
        "verdict": {"strict_correct": correct},
        "judge": {
            "backend": "openai-compatible",
            "model": finalizer.FROZEN_JUDGE_MODEL,
            "cache_key": cache_key,
            "backend_config": config,
            "backend_config_hash": config_hash,
            "error": None,
            "response_metadata": {
                "served_model": finalizer.FROZEN_JUDGE_MODEL,
            },
        },
    }


def template_row(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "candidate_answer": "placeholder",
        "subject": "Math",
        "grade": 7,
        "answer_type": "open_ended",
        "setup": "old",
        "question_text": None,
        "question_image_url": f"/remote/{task_id}-q.png",
        "reference_answer": None,
        "reference_image_url": f"/remote/{task_id}-r.png",
        "acceptable_answers": [],
        "metadata": {},
    }


class FinalizeOnlineEvaluationTest(unittest.TestCase):
    MODE = "solver_critic_repair"
    SETUP = "maxim_solver_critic_repair"
    LABEL = "solver-critic-repair-v1"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmark.jsonl"
        self.solver = self.root / "solver.jsonl"
        self.baseline_solver = self.root / "baseline_solver.jsonl"
        self.baseline_judge = self.root / "baseline_judge.jsonl"
        self.template = self.root / "template.jsonl"
        self.output = self.root / "evaluation"
        self.fresh_result = self.root / "fresh_result.jsonl"

        write_jsonl(
            self.benchmark,
            [
                {
                    "task_id": "d1",
                    "subject": "Math",
                    "answer_type": "choice",
                    "reference_answer": "B",
                },
                {
                    "task_id": "i1",
                    "subject": "Math",
                    "answer_type": "free_form",
                    "reference_answer": "http://reference.invalid/i1.png",
                },
                {
                    "task_id": "i2",
                    "subject": "Science",
                    "answer_type": "free_form",
                    "reference_answer": "http://reference.invalid/i2.png",
                },
            ],
        )
        write_jsonl(
            self.baseline_solver,
            [
                solver_row("i2", "old", "old reasoning", online=False),
                solver_row("d1", "A", "", online=False),
                solver_row("i1", "same", "same reasoning", online=False),
            ],
        )
        write_jsonl(
            self.solver,
            [
                solver_row("d1", "B", "", online=True),
                solver_row("i1", "same", "same reasoning", online=True),
                solver_row("i2", "new", "new reasoning", online=True),
            ],
        )
        write_jsonl(
            self.baseline_judge,
            [
                judge_row("d1", "exact", False, "historical"),
                judge_row("i1", "vlm_image_judge", True, "judge-v2"),
                judge_row("i2", "vlm_image_judge", False, "judge-v2"),
            ],
        )
        write_jsonl(self.template, [template_row("i1"), template_row("i2")])
        self.prepare()
        fresh_queue = preparation.read_jsonl(
            self.output / "fresh_judge_v2_input.jsonl", "fresh queue"
        )
        write_jsonl(
            self.fresh_result,
            [bound_fresh_judge_row(fresh_queue[0], True)],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self) -> None:
        finalizer.prepare_stage(
            benchmark_path=self.benchmark,
            solver_path=self.solver,
            baseline_solver_path=self.baseline_solver,
            baseline_judge_path=self.baseline_judge,
            image_template_path=self.template,
            output_dir=self.output,
            mode=self.MODE,
            setup=self.SETUP,
            label=self.LABEL,
            expected_rows=3,
            expected_deterministic=1,
            expected_image_judge=2,
            expected_benchmark_sha256=None,
            expected_baseline_solver_sha256=None,
            expected_baseline_judge_sha256=None,
            expected_image_template_sha256=None,
        )

    @contextmanager
    def fixture_protocol(self):
        patches = {
            "FROZEN_BENCHMARK_SHA256": preparation.sha256_file(self.benchmark),
            "FROZEN_BASELINE_SOLVER_SHA256": preparation.sha256_file(
                self.baseline_solver
            ),
            "FROZEN_BASELINE_JUDGE_SHA256": preparation.sha256_file(
                self.baseline_judge
            ),
            "FROZEN_IMAGE_TEMPLATE_SHA256": preparation.sha256_file(self.template),
            "DEFAULT_EXPECTED_ROWS": 3,
            "DEFAULT_EXPECTED_DETERMINISTIC": 1,
            "DEFAULT_EXPECTED_IMAGE_JUDGE": 2,
        }
        with ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(preparation, name, value))
            yield

    def finalize(self, *, overwrite: bool = False) -> dict[str, object]:
        with self.fixture_protocol():
            return finalizer.finalize_stage(
                output_dir=self.output,
                mode=self.MODE,
                setup=self.SETUP,
                label=self.LABEL,
                fresh_judge_path=self.fresh_result,
                overwrite_final=overwrite,
            )

    def test_end_to_end_prepare_merge_and_score(self) -> None:
        report = self.finalize()

        self.assertEqual(report["score"], "3/3")
        self.assertEqual(report["fresh_judge_rows"], 1)
        self.assertEqual(report["reused_judge_rows"], 1)
        score = json.loads((self.output / "score.json").read_text(encoding="utf-8"))
        self.assertEqual(score["overall"]["new_correct"], 3)
        final_manifest = json.loads(
            (self.output / "finalization_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(final_manifest["mode"], self.MODE)
        self.assertEqual(final_manifest["fresh_judge_v2_result"]["rows"], 1)
        self.assertEqual(
            final_manifest["frozen_judge_v2"]["backend_config_sha256"],
            finalizer.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
        )
        self.assertIn(
            "i2", final_manifest["frozen_judge_v2"]["request_linkage"]
        )
        self.assertTrue((self.output / "matched_image97_judge.sha256").is_file())
        self.assertTrue((self.output / "finalization.sha256").is_file())

    def test_rejects_wrong_identity_or_fresh_order(self) -> None:
        with self.fixture_protocol():
            with self.assertRaisesRegex(finalizer.FinalizationError, "mode mismatch"):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode="other",
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

        # Make both image candidates changed, rebuild into a separate output,
        # and provide the two fresh verdicts in reverse queue order.
        other_output = self.root / "other-evaluation"
        rows = preparation.read_jsonl(self.solver, "solver")
        rows[1]["solution_steps"] = "changed reasoning"
        write_jsonl(self.solver, rows)
        finalizer.prepare_stage(
            benchmark_path=self.benchmark,
            solver_path=self.solver,
            baseline_solver_path=self.baseline_solver,
            baseline_judge_path=self.baseline_judge,
            image_template_path=self.template,
            output_dir=other_output,
            mode=self.MODE,
            setup=self.SETUP,
            label=self.LABEL,
            expected_rows=3,
            expected_deterministic=1,
            expected_image_judge=2,
            expected_benchmark_sha256=None,
            expected_baseline_solver_sha256=None,
            expected_baseline_judge_sha256=None,
            expected_image_template_sha256=None,
        )
        other_queue = preparation.read_jsonl(
            other_output / "fresh_judge_v2_input.jsonl", "other fresh queue"
        )
        other_by_id = {str(row["task_id"]): row for row in other_queue}
        write_jsonl(
            self.fresh_result,
            [
                bound_fresh_judge_row(other_by_id["i2"], True),
                bound_fresh_judge_row(other_by_id["i1"], True),
            ],
        )
        with self.fixture_protocol():
            with self.assertRaisesRegex(finalizer.FinalizationError, "order differs"):
                finalizer.finalize_stage(
                    output_dir=other_output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

    def test_rejects_stale_candidate_with_same_task_ids_and_order(self) -> None:
        fresh_queue = preparation.read_jsonl(
            self.output / "fresh_judge_v2_input.jsonl", "fresh queue"
        )
        stale_queue_row = dict(fresh_queue[0])
        stale_queue_row["candidate_answer"] = "stale candidate from an older run"
        write_jsonl(
            self.fresh_result,
            [bound_fresh_judge_row(stale_queue_row, True)],
        )

        with self.fixture_protocol():
            with self.assertRaisesRegex(
                finalizer.FinalizationError,
                "request_id does not match exact prepared candidate",
            ):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

    def test_rejects_wrong_frozen_judge_settings(self) -> None:
        fresh_queue = preparation.read_jsonl(
            self.output / "fresh_judge_v2_input.jsonl", "fresh queue"
        )
        row = bound_fresh_judge_row(fresh_queue[0], True)
        row["judge"]["backend_config"]["max_tokens"] = 901
        write_jsonl(self.fresh_result, [row])

        with self.fixture_protocol():
            with self.assertRaisesRegex(
                finalizer.FinalizationError, "backend_config differs"
            ):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

    def test_concurrent_finalize_lock_fails_closed(self) -> None:
        with finalizer.ExclusiveOutputDirLock(
            self.output,
            mode=self.MODE,
            label="first-process",
        ):
            with self.fixture_protocol():
                with self.assertRaisesRegex(
                    finalizer.FinalizationError, "locked by another finalize process"
                ):
                    finalizer.finalize_stage(
                        output_dir=self.output,
                        mode=self.MODE,
                        setup=self.SETUP,
                        label=self.LABEL,
                        fresh_judge_path=self.fresh_result,
                    )
        self.assertFalse((self.output / finalizer.FINALIZE_LOCK_NAME).exists())
        self.assertFalse((self.output / "matched_image97_judge.jsonl").exists())

    def test_rejects_prompt_incompatible_fresh_result(self) -> None:
        fresh_queue = preparation.read_jsonl(
            self.output / "fresh_judge_v2_input.jsonl", "fresh queue"
        )
        row = bound_fresh_judge_row(fresh_queue[0], True)
        row["prompt_version"] = "judge-v1"
        write_jsonl(self.fresh_result, [row])
        with self.fixture_protocol():
            with self.assertRaisesRegex(finalizer.FinalizationError, "judge-v2"):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

    def test_rejects_source_or_prepared_artifact_tampering(self) -> None:
        self.template.write_text(
            self.template.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.fixture_protocol():
            with self.assertRaisesRegex(finalizer.FinalizationError, "SHA256 mismatch"):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

    def test_default_refuses_overwrite_and_explicit_flag_archives(self) -> None:
        self.finalize()
        original_score_hash = preparation.sha256_file(self.output / "score.json")
        with self.fixture_protocol():
            with self.assertRaisesRegex(finalizer.FinalizationError, "overwrite-final"):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                )

        report = self.finalize(overwrite=True)
        archive = Path(str(report["previous_outputs_archive"]))
        self.assertTrue(archive.is_dir())
        self.assertTrue((archive / "score.json").is_file())
        self.assertEqual(
            preparation.sha256_file(archive / "score.json"), original_score_hash
        )
        self.assertTrue((self.output / "score.json").is_file())

    def test_failed_explicit_overwrite_restores_previous_final(self) -> None:
        self.finalize()
        original_hash = preparation.sha256_file(self.output / "score.json")
        with self.fixture_protocol(), mock.patch.object(
            scorer,
            "build_report",
            side_effect=scorer.ScoringError("injected scorer failure"),
        ):
            with self.assertRaisesRegex(scorer.ScoringError, "injected"):
                finalizer.finalize_stage(
                    output_dir=self.output,
                    mode=self.MODE,
                    setup=self.SETUP,
                    label=self.LABEL,
                    fresh_judge_path=self.fresh_result,
                    overwrite_final=True,
                )
        self.assertEqual(preparation.sha256_file(self.output / "score.json"), original_hash)
        self.assertTrue((self.output / "finalization_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

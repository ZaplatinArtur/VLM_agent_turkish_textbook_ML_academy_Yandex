from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import finalize_maxim_online_evaluation_v1 as finalizer  # noqa: E402
import prepare_maxim_online_evaluation_v1 as preparation  # noqa: E402
import run_full274_postgeneration_v1 as pipeline  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def solver_row(
    task_id: str,
    answer: str,
    steps: str,
    *,
    online: bool,
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
    task_id: str,
    source: str,
    correct: bool,
    prompt: str,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "prompt_version": prompt,
        "metadata": {"score_source": source},
        "verdict": {"strict_correct": correct},
        "judge": {"error": None},
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


def bound_fresh_judge_row(
    queue_row: dict[str, object],
    correct: bool,
) -> dict[str, object]:
    request_id = finalizer.expected_fresh_request_id(queue_row)
    backend_config = dict(finalizer.FROZEN_JUDGE_BACKEND_CONFIG)
    backend_hash = finalizer._canonical_sha256(backend_config)
    cache_key = hashlib.sha256(
        f"{request_id}:{backend_hash}".encode("ascii")
    ).hexdigest()
    return {
        "request_id": request_id,
        "prompt_version": finalizer.FROZEN_JUDGE_PROMPT_VERSION,
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
            "backend_config": backend_config,
            "backend_config_hash": backend_hash,
            "error": None,
            "response_metadata": {
                "served_model": finalizer.FROZEN_JUDGE_MODEL,
            },
        },
    }


class Full274PostGenerationTest(unittest.TestCase):
    MODE = "active_vision"
    SETUP = "maxim_active_vision_v1"
    LABEL = "active-vision-v1"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmark.jsonl"
        self.solver = self.root / "solver.jsonl"
        self.baseline_solver = self.root / "baseline_solver.jsonl"
        self.baseline_judge = self.root / "baseline_judge.jsonl"
        self.template = self.root / "template.jsonl"
        self.output = self.root / "evaluation"

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
        self.config = pipeline.PipelineConfig(
            solver_results=self.solver.resolve(),
            setup=self.SETUP,
            output_dir=self.output.resolve(),
            base_url="http://127.0.0.1:18005/v1",
            mode=self.MODE,
            label=self.LABEL,
            benchmark=self.benchmark.resolve(),
            baseline_solver=self.baseline_solver.resolve(),
            baseline_judge=self.baseline_judge.resolve(),
            image_template=self.template.resolve(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

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
            "FROZEN_IMAGE_TEMPLATE_SHA256": preparation.sha256_file(
                self.template
            ),
            "DEFAULT_EXPECTED_ROWS": 3,
            "DEFAULT_EXPECTED_DETERMINISTIC": 1,
            "DEFAULT_EXPECTED_IMAGE_JUDGE": 2,
        }
        with ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(preparation, name, value))
            yield

    def fake_judge_runner(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        self.assertEqual(cwd, pipeline.REPO_ROOT)
        self.assertTrue(check)
        self.assertIn(str(pipeline.SRC_DIR), env["PYTHONPATH"])
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:4], ["-m", "vlm_judge.cli", "run-judge"])
        expected_pairs = {
            "--base-url": "http://127.0.0.1:18005/v1",
            "--model": finalizer.FROZEN_JUDGE_MODEL,
            "--max-tokens": "900",
            "--seed": "20260714",
            "--image-mode": "data_url",
            "--prompt-version": "judge-v2",
            "--workers": "1",
        }
        for option, expected in expected_pairs.items():
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertIn("--disable-thinking", command)
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        queue = read_jsonl(input_path)
        write_jsonl(
            output_path,
            [bound_fresh_judge_row(queue_row, True) for queue_row in queue],
        )
        return subprocess.CompletedProcess(command, 0)

    def test_plan_is_read_only_and_frozen(self) -> None:
        with self.fixture_protocol():
            report = pipeline.plan(self.config)

        self.assertFalse(report["network_or_gpu_actions_performed"])
        self.assertFalse(self.output.exists())
        self.assertEqual(
            report["judge"]["backend_config_sha256"],
            finalizer.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
        )
        command = report["stages"][1]["command"]
        self.assertIn("--disable-thinking", command)
        self.assertEqual(command[command.index("--workers") + 1], "1")

    def test_end_to_end_delegates_prepare_judge_and_finalize(self) -> None:
        with self.fixture_protocol():
            prepared = pipeline.prepare_stage(self.config)
            judged = pipeline.judge_stage(
                self.config,
                runner=self.fake_judge_runner,
            )
            finalized = pipeline.finalize_stage(self.config)

        self.assertEqual(prepared["fresh_judge_rows"], 1)
        self.assertEqual(judged["rows"], 1)
        self.assertEqual(finalized["score"], "3/3")
        state_path = self.output / pipeline.STATE_NAME
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(state["stages"]), {"prepare", "judge", "finalize"}
        )
        self.assertEqual(state["identity"]["setup"], self.SETUP)
        self.assertEqual(
            state["judge"]["backend_config_sha256"],
            finalizer.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
        )
        self.assertEqual(len(state["events"]), 3)
        checksum = (self.output / pipeline.STATE_CHECKSUM_NAME).read_text(
            encoding="ascii"
        )
        self.assertEqual(
            checksum.strip(),
            f"{pipeline.sha256_file(state_path)}  {pipeline.STATE_NAME}",
        )

    def test_resume_failure_preserves_previous_valid_judge_output(self) -> None:
        with self.fixture_protocol():
            pipeline.prepare_stage(self.config)
            pipeline.judge_stage(
                self.config,
                runner=self.fake_judge_runner,
            )
            output = self.output / pipeline.FRESH_RESULT_NAME
            original = output.read_bytes()

            def failing_runner(
                command: list[str],
                *,
                cwd: Path,
                env: dict[str, str],
                check: bool,
            ) -> subprocess.CompletedProcess[str]:
                del cwd, env, check
                temporary = Path(command[command.index("--output") + 1])
                temporary.write_text("partial\n", encoding="utf-8")
                raise subprocess.CalledProcessError(1, command)

            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.judge_stage(
                    self.config,
                    resume_judge=True,
                    runner=failing_runner,
                )

        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(list(self.output.glob(".fresh_judge_v2_result.jsonl.*.tmp")))

    def test_changed_solver_is_rejected_after_prepare(self) -> None:
        with self.fixture_protocol():
            pipeline.prepare_stage(self.config)
            with self.solver.open("a", encoding="utf-8") as destination:
                destination.write("\n")
            with self.assertRaisesRegex(
                pipeline.PostGenerationError,
                "source changed after prepare: solver",
            ):
                pipeline.judge_stage(
                    self.config,
                    runner=self.fake_judge_runner,
                )

    def test_nonfrozen_base_url_is_rejected_before_io(self) -> None:
        args = SimpleNamespace(
            solver_results=self.solver,
            setup=self.SETUP,
            output_dir=self.output,
            base_url="http://gpu.example:8000/v1",
            mode=None,
            label=None,
            benchmark=self.benchmark,
            baseline_solver=self.baseline_solver,
            baseline_judge=self.baseline_judge,
            image_template=self.template,
        )
        with self.assertRaisesRegex(
            pipeline.PostGenerationError,
            "differs from the frozen matched judge-v2 lineage",
        ):
            pipeline.PipelineConfig.from_args(args)

    def test_cli_accepts_short_solver_alias_and_defaults_identity(self) -> None:
        args = pipeline._parser().parse_args(
            [
                "plan",
                "--solver",
                str(self.solver),
                "--setup",
                self.SETUP,
                "--output-dir",
                str(self.output),
                "--base-url",
                "http://127.0.0.1:18005/v1/",
                "--benchmark",
                str(self.benchmark),
                "--baseline-solver",
                str(self.baseline_solver),
                "--baseline-judge",
                str(self.baseline_judge),
                "--image-template",
                str(self.template),
            ]
        )
        config = pipeline.PipelineConfig.from_args(args)
        self.assertEqual(config.solver_results, self.solver.resolve())
        self.assertEqual(config.mode, self.SETUP)
        self.assertEqual(config.label, self.SETUP)
        self.assertEqual(config.base_url, "http://127.0.0.1:18005/v1")


if __name__ == "__main__":
    unittest.main()

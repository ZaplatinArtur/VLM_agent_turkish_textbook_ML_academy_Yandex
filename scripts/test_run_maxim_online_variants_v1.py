from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_maxim_agent_ideas as core  # noqa: E402
import run_maxim_online_variants_v1 as variants  # noqa: E402


class FakePool:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.model = "fake/model"

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return {
            "parsed": response,
            "raw": json.dumps(response),
            "endpoint": "http://fake/v1",
            "finish_reason": "stop",
            "attempt": 1,
            "latency_s": 0.01,
            "input_tokens": 10,
            "output_tokens": 5,
            "recovered_partial": False,
            "parse_error": None,
        }


def answer(value: str) -> dict[str, str]:
    return {
        "reasoning": f"reason for {value}",
        "solution_steps": f"steps for {value}",
        "final_answer": value,
    }


class OnlineVariantsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "q.png").write_bytes(b"not-a-real-png")
        full_task = {
            "task_id": "t1",
            "subject": "Math",
            "grade": 7,
            "question": "Visible question",
            "question_images": [{"data": "nested/q.png"}],
            "answer_type": "choice",
            "reference_answer": "TOP_SECRET_GOLD",
            "reference_solution": "TOP_SECRET_SOLUTION",
        }
        self.task = core._task_view(full_task)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_solver_critic_repair_uses_three_answer_blind_calls(self) -> None:
        critique = {
            "accept_draft": False,
            "error_type": "calculation",
            "first_incorrect_step": "2+2 was copied as 5",
            "visible_evidence": "the image shows 2+2",
            "repair_instruction": "recompute",
        }
        pool = FakePool([answer("A"), critique, answer("B")])

        row = variants.run_solver_critic_repair(
            self.task,
            pool=pool,
            image_root=self.root,
            image_url_root="http://127.0.0.1:18080",
        )

        self.assertEqual(row["final_answer"], "B")
        self.assertEqual(row["generation"]["call_count"], 3)
        self.assertEqual(row["generation"]["critique"], critique)
        self.assertIsNone(row["error"])
        serialized = json.dumps(pool.requests, ensure_ascii=False)
        self.assertNotIn("TOP_SECRET", serialized)
        critic_text = pool.requests[1]["messages"][1]["content"][0]["text"]
        self.assertIn('"final_answer": "A"', critic_text)
        image_block = pool.requests[0]["messages"][1]["content"][1]
        self.assertEqual(
            image_block["image_url"]["url"],
            "http://127.0.0.1:18080/q.png",
        )

    def test_solver_critic_repair_falls_back_to_valid_draft(self) -> None:
        pool = FakePool([answer("C"), RuntimeError("critic unavailable")])

        row = variants.run_solver_critic_repair(
            self.task,
            pool=pool,
            image_root=self.root,
            image_url_root="file:///images",
        )

        self.assertEqual(row["final_answer"], "C")
        self.assertIsNone(row["error"])
        self.assertEqual(row["generation"]["fallback_stage"], "draft")
        self.assertIn("critic unavailable", row["generation"]["stage_error"])

    def test_two_pass_embeds_transcription_in_second_call(self) -> None:
        transcription = {
            "question_stem": "What is 2+2?",
            "options_or_required_parts": ["A) 3", "B) 4"],
            "critical_values_symbols_units": ["2+2"],
            "table_graph_relations": [],
            "ambiguous_spans": [],
        }
        pool = FakePool([transcription, answer("B")])

        row = variants.run_two_pass_transcription(
            self.task,
            pool=pool,
            image_root=self.root,
            image_url_root="file:///images",
        )

        self.assertEqual(row["final_answer"], "B")
        self.assertEqual(row["generation"]["transcription"], transcription)
        second_messages = json.dumps(pool.requests[1]["messages"], ensure_ascii=False)
        self.assertIn("What is 2+2?", second_messages)
        self.assertIn("original image", second_messages)

    def test_error_memory_is_frozen_and_records_checks(self) -> None:
        response = {
            "checks_applied": ["option mapping", "negative wording"],
            **answer("D"),
        }
        pool = FakePool([response])

        row = variants.run_error_memory(
            self.task,
            pool=pool,
            image_root=self.root,
            image_url_root="file:///images",
        )

        self.assertEqual(row["final_answer"], "D")
        self.assertEqual(row["generation"]["checks_applied"], response["checks_applied"])
        self.assertEqual(row["generation"]["memory_version"], "generic_error_taxonomy_v1")
        prompt = json.dumps(pool.requests[0]["messages"], ensure_ascii=False)
        self.assertIn(variants.ERROR_MEMORY, prompt)

    def test_dry_run_needs_no_endpoint_and_reports_gold_isolation(self) -> None:
        benchmark = self.root / "benchmark.jsonl"
        benchmark.write_text(
            json.dumps(
                {
                    **self.task,
                    "reference_answer": "TOP_SECRET_GOLD",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output = self.root / "out.jsonl"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            code = variants.main(
                [
                    "--mode",
                    "error_memory",
                    "--input",
                    str(benchmark),
                    "--image-root",
                    str(self.root),
                    "--output",
                    str(output),
                    "--allow-unfrozen-input",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["gold_access"])
        self.assertEqual(report["tasks"], 1)
        self.assertNotIn("TOP_SECRET", stdout.getvalue())
        self.assertFalse(output.exists())

    def test_unfrozen_benchmark_is_rejected_by_default(self) -> None:
        benchmark = self.root / "benchmark.jsonl"
        benchmark.write_text(json.dumps(self.task) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(SystemExit, "benchmark SHA256 mismatch"):
            variants.main(
                [
                    "--mode",
                    "error_memory",
                    "--input",
                    str(benchmark),
                    "--image-root",
                    str(self.root),
                    "--output",
                    str(self.root / "out.jsonl"),
                    "--dry-run",
                ]
            )

    def test_existing_output_requires_explicit_resume(self) -> None:
        benchmark = self.root / "benchmark.jsonl"
        benchmark.write_text(json.dumps(self.task) + "\n", encoding="utf-8")
        output = self.root / "out.jsonl"
        output.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "pass --resume explicitly"):
            variants.main(
                [
                    "--mode",
                    "error_memory",
                    "--input",
                    str(benchmark),
                    "--image-root",
                    str(self.root),
                    "--output",
                    str(output),
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--allow-unfrozen-input",
                ]
            )


if __name__ == "__main__":
    unittest.main()

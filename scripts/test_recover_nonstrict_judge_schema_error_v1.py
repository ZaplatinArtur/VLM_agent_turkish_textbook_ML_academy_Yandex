from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import recover_nonstrict_judge_schema_error_v1 as recovery  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def good_row(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "prompt_version": "judge-v2",
        "verdict": {"strict_correct": True},
        "judge": {"error": None},
    }


def failed_row(task_id: str, error: str = recovery.EXPECTED_ERROR) -> dict[str, object]:
    return {
        "task_id": task_id,
        "prompt_version": "judge-v2",
        "verdict": None,
        "judge": {
            "error": error,
            "attempts": 2,
            "response_metadata": {"response_id": "captured"},
        },
    }


class RecoverNonstrictJudgeSchemaErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.audit = self.root / "audit.json"
        self.schema = REPO_ROOT / "src" / "vlm_judge" / "schema.py"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_recovery(self, rows: list[dict[str, object]]) -> dict[str, object]:
        digest = write_jsonl(self.input, rows)
        return recovery.recover(
            input_path=self.input,
            output_path=self.output,
            audit_path=self.audit,
            schema_path=self.schema,
            expected_input_sha256=digest,
            expected_task_id="bad",
        )

    def test_recovers_only_provably_nonstrict_failure(self) -> None:
        audit = self.run_recovery([good_row("ok"), failed_row("bad")])
        rows = recovery.load_jsonl(self.output)
        repaired = rows[1]
        self.assertFalse(repaired["verdict"]["strict_correct"])
        self.assertEqual(repaired["verdict"]["label"], "partially_correct")
        self.assertIsNone(repaired["judge"]["error"])
        self.assertEqual(
            repaired["judge"]["schema_recovery"]["policy"],
            "fail_closed_no_positive_credit",
        )
        self.assertEqual(audit["metric_effect"]["strict_correct_credit"], 0)
        self.assertEqual(json.loads(self.audit.read_text(encoding="utf-8")), audit)

    def test_refuses_unexpected_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure set"):
            self.run_recovery([failed_row("bad", "RuntimeError: endpoint failed")])

    def test_refuses_additional_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure set"):
            self.run_recovery([failed_row("bad"), failed_row("also-bad")])

    def test_refuses_input_sha_mismatch(self) -> None:
        write_jsonl(self.input, [failed_row("bad")])
        with self.assertRaisesRegex(ValueError, "input SHA256 mismatch"):
            recovery.recover(
                input_path=self.input,
                output_path=self.output,
                audit_path=self.audit,
                schema_path=self.schema,
                expected_input_sha256="0" * 64,
                expected_task_id="bad",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("materialize_nonstrict_judge_recovery_cache_v1.py")
SPEC = importlib.util.spec_from_file_location("materialize_recovery", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MaterializeRecoveryCacheTests(unittest.TestCase):
    def _fixture(self, root: Path, *, strict_correct: bool = False):
        recovered = root / "recovered.jsonl"
        audit = root / "recovery_audit.json"
        row = {
            "request_id": "r",
            "prompt_version": "judge-v2",
            "task_id": "val_1",
            "verdict": {"label": "partially_correct", "strict_correct": strict_correct},
            "judge": {
                "backend": "openai-compatible",
                "model": "model",
                "cache_key": "a" * 64,
                "backend_config": {"backend": "openai-compatible"},
                "backend_config_hash": "b" * 64,
                "error": None,
                "response_metadata": {"served_model": "model"},
                "schema_recovery": {
                    "schema_version": module.RECOVERY_SCHEMA,
                    "policy": module.RECOVERY_POLICY,
                },
            },
        }
        recovered.write_text(json.dumps(row) + "\n", encoding="utf-8")
        recovered_sha = module.sha256_file(recovered)
        audit.write_text(
            json.dumps(
                {
                    "schema_version": module.RECOVERY_SCHEMA,
                    "policy": module.RECOVERY_POLICY,
                    "output": {"sha256": recovered_sha},
                    "recovered_task": {
                        "task_id": "val_1",
                        "inferred_strict_correct": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return recovered, audit, recovered_sha

    def test_materializes_only_non_strict_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovered, recovery_audit, recovered_sha = self._fixture(root)
            result = module.materialize(
                recovered_path=recovered,
                recovery_audit_path=recovery_audit,
                cache_dir=root / "cache",
                materialization_audit_path=root / "materialization.json",
                expected_recovered_sha256=recovered_sha,
                expected_task_id="val_1",
            )
            self.assertFalse(result["metric_effect"]["positive_credit_added"])
            self.assertTrue(Path(result["cache"]["path"]).is_file())

    def test_rejects_positive_credit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovered, recovery_audit, recovered_sha = self._fixture(
                root, strict_correct=True
            )
            with self.assertRaisesRegex(ValueError, "positive strict credit"):
                module.materialize(
                    recovered_path=recovered,
                    recovery_audit_path=recovery_audit,
                    cache_dir=root / "cache",
                    materialization_audit_path=root / "materialization.json",
                    expected_recovered_sha256=recovered_sha,
                    expected_task_id="val_1",
                )


if __name__ == "__main__":
    unittest.main()

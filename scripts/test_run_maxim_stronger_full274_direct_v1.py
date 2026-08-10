from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_maxim_stronger_full274_direct_v1.py")
SPEC = importlib.util.spec_from_file_location("stronger_full274", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class StrongerFull274ProfileTests(unittest.TestCase):
    def test_profile_selects_every_task_with_frozen_27b_decoding(self) -> None:
        runner = profile.runner
        self.assertEqual(runner.MODEL, "Qwen/Qwen3.5-27B")
        self.assertEqual(runner.TEMPERATURE, 0.0)
        self.assertEqual(runner.MAX_TOKENS, 3072)
        self.assertFalse(runner.PRIMARY_ENABLE_THINKING)
        self.assertEqual(runner.EXPECTED_NATIVE_ROWS, 274)
        self.assertTrue(
            runner.is_native_hard_case(
                {"task_id": "x", "subject": "Physics"}, {}, {}
            )
        )
        self.assertEqual(
            runner.CONDITION, "maxim_stronger_27b_direct_full274_v1"
        )

    def test_true_math_detector_is_preserved_for_benchmark_audit(self) -> None:
        self.assertTrue(profile._subject_is_math({"subject": "Math"}))
        self.assertFalse(profile._subject_is_math({"subject": "Physics"}))


if __name__ == "__main__":
    unittest.main()

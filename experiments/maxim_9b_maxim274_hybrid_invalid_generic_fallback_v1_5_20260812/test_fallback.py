from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fallback_compose as fallback


class FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = fallback._load_v14()

    @staticmethod
    def prediction(**changes):
        value = {
            "schema_version": "maxim256-hybrid-generic-prediction-v1",
            "task_id": "x",
            "final_answer": "A",
            "option_label": "A",
            "answer_type": "choice",
            "input_mode": "ocr_only",
            "error": None,
            "generation": {
                "gold_access": False,
                "outcome_access": False,
                "model": "qwen/qwen3.5-9b",
                "provider": "SiliconFlow",
                "quantization": "fp8",
            },
        }
        value.update(changes)
        return value

    def test_same_validity_and_projection_contract(self):
        prediction = self.prediction(final_answer="visible text", option_label="C")
        self.assertTrue(self.legacy._valid_prediction(prediction, "x", "choice"))
        self.assertEqual(self.legacy._official_answer(prediction, "choice"), "C")
        numeric = self.prediction(
            final_answer=" 42 kg ", option_label="NA", answer_type="numeric"
        )
        self.assertEqual(self.legacy._official_answer(numeric, "numeric"), "42 kg")

    def test_hostile_schema_and_labels_fallback(self):
        for hostile in (None, [], 1, "AB", "", "a", "AA"):
            self.assertFalse(
                self.legacy._valid_prediction(
                    self.prediction(option_label=hostile), "x", "choice"
                )
            )
        self.assertFalse(
            self.legacy._valid_prediction(
                self.prediction(error="CandidateError", final_answer=""), "x", "choice"
            )
        )

    def test_full_identity_partition_and_duplicate_abort(self):
        base = [f"id-{index}" for index in range(274)]
        alignment = base[18:]
        decisions = [
            {
                "runtime_alignment_id": task_id,
                "branch": "certified_noid" if index < 18 else "generic_qwen35_9b",
            }
            for index, task_id in enumerate(base)
        ]
        self.assertEqual(
            self.legacy._validate_identity_closure(decisions, base, alignment),
            set(alignment),
        )
        decisions[-1]["runtime_alignment_id"] = decisions[-2]["runtime_alignment_id"]
        with self.assertRaises(self.legacy.FallbackError):
            self.legacy._validate_identity_closure(decisions, base, alignment)

    def test_windows_hostile_terminal_lf_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "output.jsonl"
            expected = b'{"task_id":"x"}\n'
            fallback.exclusive_bytes(path, expected)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(path.read_bytes().count(b"\r"), 0)
            self.assertEqual(fallback.sha256(path), fallback.sha256_bytes(expected))

    def test_jsonl_rejects_crlf_so_raw_claim_is_unambiguous(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.jsonl"
            path.write_bytes(b'{"task_id":"x"}\r\n')
            with self.assertRaises(fallback.FallbackError):
                fallback.jsonl_with_raw(path)

    def test_actual_base240_is_lf_only_and_pinned(self):
        data = fallback.stable_bytes(fallback.BASE240)
        self.assertEqual(fallback.sha256(fallback.BASE240), fallback.BASE240_SHA)
        self.assertEqual(data.count(b"\r"), 0)
        self.assertEqual(data.count(b"\n"), 274)
        self.assertEqual(len(fallback.jsonl_with_raw(fallback.BASE240)), 274)

    def test_pre_outcome_outputs_are_absent(self):
        self.assertFalse(fallback.COMPLETION.exists())
        self.assertFalse(fallback.PREDICTIONS.exists())
        self.assertFalse(fallback.OUTPUT.exists())
        self.assertFalse(fallback.MANIFEST.exists())


if __name__ == "__main__":
    unittest.main()

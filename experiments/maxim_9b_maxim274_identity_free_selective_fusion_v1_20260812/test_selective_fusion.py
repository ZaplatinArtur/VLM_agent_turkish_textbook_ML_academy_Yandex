from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import selective_fusion as fusion


class FakeNormalizer:
    @staticmethod
    def normalize_multiple_choice(value: str) -> str | None:
        return value if value in "ABCDE" and len(value) == 1 else None

    @staticmethod
    def parse_numeric(value: str) -> object | None:
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.strip().casefold()


def prediction(*, answer_type: str = "choice", answer: str = "alpha", option: str = "A", error=None) -> dict:
    return {
        "schema_version": "maxim256-hybrid-generic-prediction-v1",
        "task_id": "outer-only",
        "final_answer": answer,
        "option_label": option,
        "answer_type": answer_type,
        "input_mode": "ocr_only",
        "error": error,
        "generation": {
            "gold_access": False,
            "outcome_access": False,
            "model": "qwen/qwen3.5-9b",
            "provider": "SiliconFlow",
            "quantization": "fp8",
        },
    }


class SelectiveFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = FakeNormalizer()

    def select(self, baseline: str, row: dict, answer_type: str = "choice") -> dict:
        projection = fusion.generic_projection(row, "outer-only", answer_type)
        self.assertNotIn("task_id", projection)
        return fusion.select_observable(answer_type=answer_type, baseline_answer=baseline, generic_projection=projection, normalizer=self.normalizer)

    def test_parseable_baseline_is_held_even_on_disagreement(self) -> None:
        self.assertEqual(self.select("B", prediction(option="A"))["selected"], "baseline")

    def test_unparseable_baseline_allows_valid_generic(self) -> None:
        action = self.select("not a label", prediction(option="C"))
        self.assertEqual(action["selected"], "generic")
        self.assertEqual(action["reason"], "valid_generic_replaces_unparseable_baseline")

    def test_invalid_generic_always_falls_back(self) -> None:
        self.assertEqual(self.select("not a label", prediction(error="timeout"))["selected"], "baseline")
        hostile = prediction()
        hostile["unexpected"] = True
        self.assertEqual(self.select("not a label", hostile)["selected"], "baseline")

    def test_numeric_and_text_baseline_parseability(self) -> None:
        numeric = prediction(answer_type="numeric", answer="4", option="NA")
        self.assertEqual(self.select("4.0", numeric, "numeric")["selected"], "baseline")
        self.assertEqual(self.select("four", numeric, "numeric")["selected"], "generic")
        text = prediction(answer_type="short_text", answer="replacement", option="NA")
        self.assertEqual(self.select("existing text", text, "short_text")["selected"], "baseline")

    def test_selector_has_no_identity_parameter(self) -> None:
        import inspect
        self.assertEqual(set(inspect.signature(fusion.select_observable).parameters), {"answer_type", "baseline_answer", "generic_projection", "normalizer"})

    def test_identity_mismatch_aborts_before_selector(self) -> None:
        with self.assertRaises(fusion.FusionError):
            fusion.generic_projection(prediction(), "different-id", "choice")

    def test_binary_exclusive_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "x.jsonl"
            fusion.exclusive_bytes(path, b'{"x":1}\n')
            self.assertEqual(path.read_bytes(), b'{"x":1}\n')
            with self.assertRaises(FileExistsError):
                fusion.exclusive_bytes(path, b"overwrite")


if __name__ == "__main__":
    unittest.main()

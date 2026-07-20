import unittest

from vlm_judge.metrics import deterministic_match


class DeterministicMetricsTests(unittest.TestCase):
    def test_multiple_choice(self) -> None:
        result = deterministic_match("C", "### Answer\nC) 42", "multiple_choice")
        self.assertTrue(result.applicable)
        self.assertTrue(result.matched)

    def test_final_heading_beats_boxed_expression_in_reasoning(self) -> None:
        candidate = "Reasoning uses \\boxed{2}.\n\n### Answer\n\nB) final choice"
        result = deterministic_match("B", candidate, "multiple_choice")
        self.assertTrue(result.applicable)
        self.assertTrue(result.matched)

    def test_equivalent_fraction_and_decimal(self) -> None:
        result = deterministic_match("1/2", "Answer: 0.5", "numeric")
        self.assertTrue(result.applicable)
        self.assertTrue(result.matched)

    def test_non_equivalent_number(self) -> None:
        result = deterministic_match("12", "13", "numeric")
        self.assertFalse(result.matched)

    def test_unknown_requires_judge(self) -> None:
        result = deterministic_match("Paris", "The answer is Paris", "open_ended")
        self.assertFalse(result.applicable)
        self.assertIsNone(result.matched)


if __name__ == "__main__":
    unittest.main()

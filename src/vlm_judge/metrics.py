from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .normalization import normalize_multiple_choice, normalize_text, parse_numeric


@dataclass(frozen=True, slots=True)
class DeterministicResult:
    applicable: bool
    matched: bool | None
    method: str
    normalized_reference: str | None = None
    normalized_candidate: str | None = None


def _numeric_close(reference: Fraction, candidate: Fraction, tolerance: float) -> bool:
    return abs(float(reference - candidate)) <= tolerance


def deterministic_match(
    reference: str | None,
    candidate: str,
    answer_type: str,
    *,
    acceptable_answers: list[str] | None = None,
    numeric_tolerance: float = 1e-9,
) -> DeterministicResult:
    if not reference:
        return DeterministicResult(False, None, "no_text_reference")

    references = [reference, *(acceptable_answers or [])]
    if answer_type == "multiple_choice":
        candidate_value = normalize_multiple_choice(candidate)
        reference_values = [normalize_multiple_choice(x) for x in references]
        if candidate_value is None or all(x is None for x in reference_values):
            return DeterministicResult(False, None, "multiple_choice_parse_failed")
        matched = candidate_value in reference_values
        return DeterministicResult(
            True,
            matched,
            "multiple_choice",
            ",".join(x for x in reference_values if x),
            candidate_value,
        )

    if answer_type == "numeric":
        candidate_value = parse_numeric(candidate)
        reference_values = [parse_numeric(x) for x in references]
        valid_references = [x for x in reference_values if x is not None]
        if candidate_value is None or not valid_references:
            return DeterministicResult(False, None, "numeric_parse_failed")
        matched = any(
            _numeric_close(reference_value, candidate_value, numeric_tolerance)
            for reference_value in valid_references
        )
        return DeterministicResult(
            True,
            matched,
            "numeric",
            ",".join(str(x) for x in valid_references),
            str(candidate_value),
        )

    if answer_type == "short_text":
        normalized_candidate = normalize_text(candidate)
        normalized_references = [normalize_text(x) for x in references]
        return DeterministicResult(
            True,
            normalized_candidate in normalized_references,
            "normalized_text",
            " | ".join(normalized_references),
            normalized_candidate,
        )

    return DeterministicResult(False, None, "llm_judge_required")


"""Frozen content-only failover rule for the 185-row YKSLOP DEV split.

The primary arm is used only when its persisted row is an exact, successful
V6.2 record with a valid A-E answer.  Every other condition falls back to the
already-completed V5 theory answer.  No confidence, task identifier, outcome,
or correctness signal is accepted by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


ANSWERS = frozenset("ABCDE")
V6_SCHEMA = "generic-medium-nonstream-content-prediction-v6"
FALLBACK_SCHEMA = "generic-v5-theory-content-fallback-v1"
OUTPUT_SCHEMA = "generic-v6-v5-theory-failover-prediction-v1"

V6_KEYS = frozenset(
    {
        "schema_version",
        "content_sha256",
        "request_sha256",
        "prediction",
        "terminal_success",
        "attempt_count",
        "terminal_error_kind",
        "model_contract_error",
        "gold_access",
        "final_access",
        "opaque_identifier_access",
    }
)
FALLBACK_KEYS = frozenset(
    {
        "schema_version",
        "content_sha256",
        "prediction",
        "source_arm",
        "gold_access",
        "final_access",
        "opaque_identifier_retained",
    }
)


class FailoverError(RuntimeError):
    pass


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_fallback_row(row: Any) -> dict[str, Any]:
    if type(row) is not dict or frozenset(row) != FALLBACK_KEYS:
        raise FailoverError("fallback schema mismatch")
    if (
        row.get("schema_version") != FALLBACK_SCHEMA
        or not _is_sha256(row.get("content_sha256"))
        or row.get("prediction") not in ANSWERS
        or row.get("source_arm") != "local_textbook_theory_bm25"
        or row.get("gold_access") is not False
        or row.get("final_access") is not False
        or row.get("opaque_identifier_retained") is not False
    ):
        raise FailoverError("invalid fallback row")
    return dict(row)


def is_strict_valid_v6(row: Any, expected_content_sha256: str) -> bool:
    """Return true only for an exact successful V6.2 prediction contract."""

    return bool(
        type(row) is dict
        and frozenset(row) == V6_KEYS
        and row.get("schema_version") == V6_SCHEMA
        and row.get("content_sha256") == expected_content_sha256
        and _is_sha256(row.get("request_sha256"))
        and row.get("prediction") in ANSWERS
        and row.get("terminal_success") is True
        and type(row.get("attempt_count")) is int
        and 1 <= row["attempt_count"] <= 3
        and row.get("terminal_error_kind") is None
        and row.get("model_contract_error") is None
        and row.get("gold_access") is False
        and row.get("final_access") is False
        and row.get("opaque_identifier_access") is False
    )


def apply_failover(
    v6_rows: list[Any], fallback_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the preregistered row-local failover without inspecting content."""

    validated_fallbacks = [validate_fallback_row(row) for row in fallback_rows]
    fallback_hashes = [row["content_sha256"] for row in validated_fallbacks]
    if len(fallback_hashes) != 185 or len(set(fallback_hashes)) != 185:
        raise FailoverError("fallback denominator/content uniqueness mismatch")

    candidates: dict[str, list[Any]] = defaultdict(list)
    for row in v6_rows:
        if type(row) is dict and _is_sha256(row.get("content_sha256")):
            candidates[row["content_sha256"]].append(row)

    outputs: list[dict[str, Any]] = []
    for fallback in validated_fallbacks:
        content_hash = fallback["content_sha256"]
        matches = candidates.get(content_hash, [])
        if len(matches) == 0:
            answer = fallback["prediction"]
            source = "v5_theory_fallback"
            reason: str | None = "v6_missing"
        elif len(matches) > 1:
            answer = fallback["prediction"]
            source = "v5_theory_fallback"
            reason = "v6_duplicate_content"
        elif is_strict_valid_v6(matches[0], content_hash):
            answer = matches[0]["prediction"]
            source = "v6_2_strict_success"
            reason = None
        else:
            answer = fallback["prediction"]
            source = "v5_theory_fallback"
            reason = "v6_invalid_schema_or_error"
        outputs.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "content_sha256": content_hash,
                "prediction": answer,
                "selected_source": source,
                "fallback_reason": reason,
                "gold_access": False,
                "final_access": False,
                "opaque_identifier_access": False,
            }
        )
    return outputs


def selection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["selected_source"] for row in rows)
    return {key: counts[key] for key in sorted(counts)}

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from itertools import permutations
from pathlib import Path
from typing import Any

from evidence_os import (
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
    FrozenProfile,
    InferenceBundle,
    ProblemInput,
    decide,
    issue_certificate,
)
from evidence_os.ingest import PolicyCase, align_candidate_runs, load_candidate_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _visible_payload(question: str, answer: str) -> dict[str, Any]:
    return {
        "question": question,
        "answer_type": "choice",
        "final_answer": answer,
        "generation": {"gold_access": False, "call_count": 1},
    }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(child) for child in value]
    return value


def _policy_projection(case: PolicyCase) -> dict[str, dict[str, Any]]:
    return {
        source: _thaw(payload)
        for source, payload in case.candidates.items()
    }


def test_policy_contracts_have_no_benchmark_identifier_field() -> None:
    assert {field.name for field in fields(ProblemInput)} == {
        "statement",
        "image_fingerprints",
        "constraints",
        "answer_format",
    }
    assert "task_id" not in {field.name for field in fields(InferenceBundle)}
    assert {field.name for field in fields(PolicyCase)} == {"candidates"}


def test_randomizing_alignment_ids_does_not_change_policy_visible_cases(
    tmp_path: Path,
) -> None:
    original_anchor = tmp_path / "anchor_original.jsonl"
    original_challenger = tmp_path / "challenger_original.jsonl"
    randomized_anchor = tmp_path / "anchor_randomized.jsonl"
    randomized_challenger = tmp_path / "challenger_randomized.jsonl"

    original_ids = ("validation-row-0001", "validation-row-0002")
    randomized_ids = ("opaque-f47ae0", "opaque-91c26b")
    payloads = (
        (_visible_payload("2 + 2 = ?", "C"), _visible_payload("2 + 2 = ?", "B")),
        (_visible_payload("3 + 3 = ?", "A"), _visible_payload("3 + 3 = ?", "A")),
    )

    for anchor_path, challenger_path, task_ids in (
        (original_anchor, original_challenger, original_ids),
        (randomized_anchor, randomized_challenger, randomized_ids),
    ):
        _write_jsonl(
            anchor_path,
            [
                {"task_id": task_id, **anchor_payload}
                for task_id, (anchor_payload, _) in zip(task_ids, payloads, strict=True)
            ],
        )
        _write_jsonl(
            challenger_path,
            [
                {"task_id": task_id, **challenger_payload}
                for task_id, (_, challenger_payload) in zip(task_ids, payloads, strict=True)
            ],
        )

    original = align_candidate_runs(
        (
            load_candidate_jsonl(original_anchor, name="anchor"),
            load_candidate_jsonl(original_challenger, name="challenger"),
        )
    )
    randomized = align_candidate_runs(
        (
            load_candidate_jsonl(randomized_anchor, name="anchor"),
            load_candidate_jsonl(randomized_challenger, name="challenger"),
        )
    )

    original_visible = [_policy_projection(case) for case in original.cases]
    randomized_visible = [_policy_projection(case) for case in randomized.cases]
    assert original_visible == randomized_visible
    assert "task_id" not in json.dumps(original_visible, ensure_ascii=False)
    assert "validation-row" not in repr(original.cases)


def test_row_permutation_changes_only_output_order() -> None:
    rows: list[InferenceBundle] = []
    for value in ("2 + 2", "3 + 3", "4 + 4"):
        problem = ProblemInput(statement=value, answer_format="choice")
        anchor = CandidateEnvelope(source="anchor", final_answer="A")
        challenger = CandidateEnvelope(source="calculator", final_answer="B")
        certificate = issue_certificate(
            problem,
            challenger,
            kind=CertificateKind.EXECUTABLE_CHECK,
            strength=CertificateStrength.STRONG,
            verdict=CertificateVerdict.PASS,
            verifier="bounded-calculator-v1",
            claim_coverage=1.0,
            contradiction_count=0,
            deterministic_checks=(True,),
            trace=f"{value} -> B",
        )
        rows.append(
            InferenceBundle(
                problem=problem,
                anchor=anchor,
                candidates=(
                    CandidateEnvelope(
                        source=challenger.source,
                        final_answer=challenger.final_answer,
                        certificates=(certificate,),
                    ),
                ),
            )
        )

    def decisions(bundles: tuple[InferenceBundle, ...]) -> dict[str, tuple[str, str]]:
        return {
            bundle.problem.statement: (
                decide(bundle, FrozenProfile()).action.value,
                decide(bundle, FrozenProfile()).selected.final_answer,
            )
            for bundle in bundles
        }

    expected = decisions(tuple(rows))
    for order in permutations(rows):
        assert decisions(order) == expected

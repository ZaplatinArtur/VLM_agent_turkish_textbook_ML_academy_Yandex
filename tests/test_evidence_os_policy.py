from __future__ import annotations

from itertools import permutations
from typing import cast

import pytest

from evidence_os import (
    CandidateEnvelope,
    Certificate,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
    DecisionAction,
    DecisionReason,
    FrozenProfile,
    InferenceBundle,
    ProblemInput,
    decide,
    issue_certificate,
)


def _problem() -> ProblemInput:
    return ProblemInput(
        statement="120 sayisinin yuzde 15'i kactir?",
        constraints=("Select exactly one option",),
        answer_format="choice",
    )


def _anchor() -> CandidateEnvelope:
    return CandidateEnvelope(source="frozen-anchor", final_answer="C")


def _strong_certificate(
    problem: ProblemInput,
    answer: CandidateEnvelope | str,
    *,
    verifier: str = "bounded-calculator-v1",
    checks: tuple[bool, ...] = (True,),
) -> Certificate:
    return issue_certificate(
        problem,
        answer,
        kind=CertificateKind.EXECUTABLE_CHECK,
        strength=CertificateStrength.STRONG,
        verdict=CertificateVerdict.PASS,
        verifier=verifier,
        claim_coverage=1.0,
        contradiction_count=0,
        deterministic_checks=checks,
        trace="120 * 15 / 100 = 18; option 18 -> B",
    )


def test_empty_citations_and_no_certificate_cannot_override_anchor() -> None:
    problem = _problem()
    challenger = CandidateEnvelope(source="web-with-empty-citations", final_answer="B")

    decision = decide(
        InferenceBundle(problem=problem, anchor=_anchor(), candidates=(challenger,))
    )

    assert decision.action is DecisionAction.KEEP_ANCHOR
    assert decision.selected == _anchor()
    assert decision.reason is DecisionReason.NO_STRONG_CERTIFICATE
    assert decision.admitted_certificates == ()


def test_calculator_answer_mismatch_rejects_certificate_and_falls_back() -> None:
    problem = _problem()
    certificate_for_a_different_answer = _strong_certificate(problem, "B")
    challenger = CandidateEnvelope(
        source="calculator",
        final_answer="D",
        certificates=(certificate_for_a_different_answer,),
    )

    decision = decide(
        InferenceBundle(problem=problem, anchor=_anchor(), candidates=(challenger,))
    )

    assert decision.action is DecisionAction.KEEP_ANCHOR
    assert decision.selected == _anchor()
    assert decision.reason is DecisionReason.NO_STRONG_CERTIFICATE


def test_failed_calculator_check_rejects_certificate_and_falls_back() -> None:
    problem = _problem()
    challenger_without_certificate = CandidateEnvelope(source="calculator", final_answer="B")
    mismatch = _strong_certificate(
        problem,
        challenger_without_certificate,
        checks=(True, False),
    )
    challenger = CandidateEnvelope(
        source="calculator",
        final_answer="B",
        certificates=(mismatch,),
    )

    decision = decide(
        InferenceBundle(problem=problem, anchor=_anchor(), candidates=(challenger,))
    )

    assert decision.action is DecisionAction.KEEP_ANCHOR
    assert decision.reason is DecisionReason.NO_STRONG_CERTIFICATE


def test_missing_or_malformed_certificate_fails_closed() -> None:
    problem = _problem()
    missing = CandidateEnvelope(source="missing-certificate", final_answer="B")
    malformed = CandidateEnvelope(
        source="malformed-certificate",
        final_answer="B",
        certificates=cast(tuple[Certificate, ...], ({"kind": "executable_check"},)),
    )

    missing_decision = decide(
        InferenceBundle(problem=problem, anchor=_anchor(), candidates=(missing,))
    )
    malformed_decision = decide(
        InferenceBundle(problem=problem, anchor=_anchor(), candidates=(malformed,))
    )

    assert missing_decision.action is DecisionAction.KEEP_ANCHOR
    assert missing_decision.reason is DecisionReason.NO_STRONG_CERTIFICATE
    assert malformed_decision.action is DecisionAction.KEEP_ANCHOR
    assert malformed_decision.reason is DecisionReason.NO_STRONG_CERTIFICATE


def test_strong_pass_without_trace_is_rejected_before_policy() -> None:
    with pytest.raises(ValueError, match="requires a verifier trace"):
        issue_certificate(
            _problem(),
            "B",
            kind=CertificateKind.EXECUTABLE_CHECK,
            strength=CertificateStrength.STRONG,
            verdict=CertificateVerdict.PASS,
            verifier="bounded-calculator-v1",
            claim_coverage=1.0,
            contradiction_count=0,
            deterministic_checks=(True,),
        )


def test_candidate_permutation_cannot_change_a_certified_decision() -> None:
    problem = _problem()
    verified_b = CandidateEnvelope(source="calculator", final_answer="B")
    certificate = _strong_certificate(problem, verified_b)
    candidates = (
        CandidateEnvelope(
            source="calculator",
            final_answer="B",
            certificates=(certificate,),
        ),
        CandidateEnvelope(source="second-solver", final_answer="  B "),
        CandidateEnvelope(source="invalid-solver", final_answer="D", valid_format=False),
    )

    expected: tuple[str, str, str, str, tuple[str, ...]] | None = None
    for order in permutations(candidates):
        decision = decide(
            InferenceBundle(problem=problem, anchor=_anchor(), candidates=order),
            FrozenProfile(),
        )
        observed = (
            decision.action.value,
            decision.reason.value,
            decision.selected.final_answer.strip(),
            decision.selected.source,
            tuple(certificate.verifier for certificate in decision.admitted_certificates),
        )
        expected = observed if expected is None else expected
        assert observed == expected

    assert expected == (
        DecisionAction.REPLACE_ANCHOR.value,
        DecisionReason.STRONGLY_VERIFIED_CHALLENGER.value,
        "B",
        "calculator",
        ("bounded-calculator-v1",),
    )

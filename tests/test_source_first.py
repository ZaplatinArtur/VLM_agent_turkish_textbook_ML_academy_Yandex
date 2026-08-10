from __future__ import annotations

import pytest

from evidence_os import (
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
    FrozenProfile,
    InferenceBundle,
    ProblemInput,
    SourceFirstAction,
    SourceFirstReason,
    decide,
    decide_source_first,
    issue_certificate,
)


def _problem() -> ProblemInput:
    return ProblemInput(statement="2 + 2 = ?", answer_format="integer")


def _certified(problem: ProblemInput, answer: str, *, verifier: str = "official-key") -> CandidateEnvelope:
    candidate = CandidateEnvelope(source=verifier, final_answer=answer)
    certificate = issue_certificate(
        problem,
        candidate,
        kind=CertificateKind.SOURCE_ENTAILMENT,
        strength=CertificateStrength.STRONG,
        verdict=CertificateVerdict.PASS,
        verifier=verifier,
        claim_coverage=1.0,
        contradiction_count=0,
        deterministic_checks=(True, True),
        trace=f"{verifier}:{answer}",
    )
    return CandidateEnvelope(
        source=verifier,
        final_answer=answer,
        certificates=(certificate,),
    )


def test_unique_certified_source_can_skip_unverified_anchor() -> None:
    problem = _problem()
    source = _certified(problem, "4")

    shortcut = decide_source_first(
        problem,
        (source,),
        anchor_may_emit_strong_certificates=False,
    )

    assert shortcut.action is SourceFirstAction.RETURN_CERTIFIED_SOURCE
    assert shortcut.selected == source
    assert shortcut.anchor_required is False

    for anchor_answer in ("4", "5"):
        full = decide(
            InferenceBundle(
                problem=problem,
                anchor=CandidateEnvelope(source="reasoning", final_answer=anchor_answer),
                candidates=(source,),
            )
        )
        assert full.selected.final_answer == shortcut.selected.final_answer


def test_missing_or_weak_source_runs_anchor() -> None:
    problem = _problem()
    weak = CandidateEnvelope(source="retrieval", final_answer="4")

    decision = decide_source_first(
        problem,
        (weak,),
        anchor_may_emit_strong_certificates=False,
    )

    assert decision.action is SourceFirstAction.RUN_ANCHOR
    assert decision.reason is SourceFirstReason.NO_CERTIFIED_SOURCE


def test_conflicting_certified_sources_run_anchor() -> None:
    problem = _problem()

    decision = decide_source_first(
        problem,
        (_certified(problem, "4", verifier="key-a"), _certified(problem, "5", verifier="key-b")),
        anchor_may_emit_strong_certificates=False,
    )

    assert decision.action is SourceFirstAction.RUN_ANCHOR
    assert decision.reason is SourceFirstReason.CONFLICTING_CERTIFIED_SOURCES
    assert len(decision.admitted_certificates) == 2


def test_anchor_certificate_capability_disables_shortcut() -> None:
    problem = _problem()

    decision = decide_source_first(
        problem,
        (_certified(problem, "4"),),
        profile=FrozenProfile(),
        anchor_may_emit_strong_certificates=True,
    )

    assert decision.action is SourceFirstAction.RUN_ANCHOR
    assert decision.reason is SourceFirstReason.ANCHOR_MAY_EMIT_STRONG_CERTIFICATES


def test_anchor_certificate_capability_must_be_explicit() -> None:
    problem = _problem()
    with pytest.raises(TypeError, match="anchor_may_emit_strong_certificates"):
        decide_source_first(problem, (_certified(problem, "4"),))

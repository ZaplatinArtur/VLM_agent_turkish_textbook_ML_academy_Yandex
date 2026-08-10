"""Canonical bindings and constructors for verifier certificates."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from .contracts import (
    CandidateEnvelope,
    Certificate,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
    ProblemInput,
)

_WHITESPACE = re.compile(r"\s+")


def canonicalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def input_fingerprint(problem: ProblemInput) -> str:
    payload = {
        "answer_format": canonicalize_text(problem.answer_format or ""),
        "constraints": [canonicalize_text(item) for item in problem.constraints],
        "image_fingerprints": sorted(problem.image_fingerprints),
        "statement": canonicalize_text(problem.statement),
    }
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def answer_fingerprint(answer: CandidateEnvelope | str) -> str:
    value = answer.final_answer if isinstance(answer, CandidateEnvelope) else answer
    return _sha256(canonicalize_text(value).encode("utf-8"))


def trace_fingerprint(trace: str | bytes) -> str:
    value = trace.encode("utf-8") if isinstance(trace, str) else trace
    if not value:
        raise ValueError("verifier trace must be non-empty")
    return _sha256(value)


def certificate_fingerprint(certificate: Certificate) -> str:
    payload = {
        "answer": certificate.answer_fingerprint,
        "answer_bound": certificate.answer_bound,
        "checks": certificate.deterministic_checks,
        "contradictions": certificate.contradiction_count,
        "coverage": certificate.claim_coverage,
        "input": certificate.input_fingerprint,
        "input_bound": certificate.input_bound,
        "kind": certificate.kind.value,
        "strength": certificate.strength.value,
        "trace": certificate.trace_fingerprint or "",
        "verdict": certificate.verdict.value,
        "verifier": certificate.verifier,
    }
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def issue_certificate(
    problem: ProblemInput,
    answer: CandidateEnvelope | str,
    *,
    kind: CertificateKind,
    strength: CertificateStrength,
    verdict: CertificateVerdict,
    verifier: str,
    claim_coverage: float,
    contradiction_count: int,
    deterministic_checks: tuple[bool, ...],
    trace: str | bytes | None = None,
) -> Certificate:
    """Issue a certificate with bindings derived from observable content."""

    return Certificate(
        kind=kind,
        strength=strength,
        verdict=verdict,
        input_fingerprint=input_fingerprint(problem),
        answer_fingerprint=answer_fingerprint(answer),
        input_bound=True,
        answer_bound=True,
        claim_coverage=claim_coverage,
        contradiction_count=contradiction_count,
        deterministic_checks=deterministic_checks,
        verifier=verifier,
        trace_fingerprint=trace_fingerprint(trace) if trace is not None else None,
    )


def certificate_matches(
    certificate: Certificate,
    problem: ProblemInput,
    answer: CandidateEnvelope | str,
) -> bool:
    return (
        certificate.input_bound
        and certificate.answer_bound
        and certificate.input_fingerprint == input_fingerprint(problem)
        and certificate.answer_fingerprint == answer_fingerprint(answer)
    )

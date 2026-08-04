"""Identifier-free contracts for evidence-gated inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CertificateKind(str, Enum):
    EXECUTABLE_CHECK = "executable_check"
    FORMAL_EQUIVALENCE = "formal_equivalence"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    SOURCE_ENTAILMENT = "source_entailment"
    VISUAL_GROUNDING = "visual_grounding"
    AGREEMENT = "agreement"
    SELF_CONFIDENCE = "self_confidence"


class CertificateStrength(str, Enum):
    WEAK = "weak"
    STRONG = "strong"


class CertificateVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


WEAK_ONLY_CERTIFICATE_KINDS = frozenset(
    {CertificateKind.AGREEMENT, CertificateKind.SELF_CONFIDENCE}
)


@dataclass(frozen=True, slots=True)
class ProblemInput:
    """Observable problem content; benchmark row identifiers are absent."""

    statement: str
    image_fingerprints: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    answer_format: str | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip() and not self.image_fingerprints:
            raise ValueError("a problem needs text or at least one image fingerprint")
        if any(not value.strip() for value in self.image_fingerprints):
            raise ValueError("image fingerprints must be non-empty")


@dataclass(frozen=True, slots=True)
class Certificate:
    """A verifier result explicitly bound to an input and an answer."""

    kind: CertificateKind
    strength: CertificateStrength
    verdict: CertificateVerdict
    input_fingerprint: str
    answer_fingerprint: str
    input_bound: bool
    answer_bound: bool
    claim_coverage: float
    contradiction_count: int
    deterministic_checks: tuple[bool, ...]
    verifier: str
    trace_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.kind in WEAK_ONLY_CERTIFICATE_KINDS and self.strength is CertificateStrength.STRONG:
            raise ValueError(f"{self.kind.value} evidence can never be strong")
        if not 0.0 <= self.claim_coverage <= 1.0:
            raise ValueError("claim_coverage must be between 0 and 1")
        if self.contradiction_count < 0:
            raise ValueError("contradiction_count cannot be negative")
        if not self.verifier.strip():
            raise ValueError("certificate verifier must be named")
        if self.input_bound and not self.input_fingerprint.strip():
            raise ValueError("input-bound certificate needs an input fingerprint")
        if self.answer_bound and not self.answer_fingerprint.strip():
            raise ValueError("answer-bound certificate needs an answer fingerprint")
        if (
            self.strength is CertificateStrength.STRONG
            and self.verdict is CertificateVerdict.PASS
            and not (self.trace_fingerprint and self.trace_fingerprint.strip())
        ):
            raise ValueError("a strong passing certificate requires a verifier trace")


@dataclass(frozen=True, slots=True)
class CandidateEnvelope:
    """One solver proposal and the evidence produced for that proposal."""

    source: str
    final_answer: str
    valid_format: bool = True
    error: str | None = None
    abstain: bool = False
    certificates: tuple[Certificate, ...] = ()
    self_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("candidate source must be non-empty")
        if not self.final_answer.strip() and not (self.abstain or self.error):
            raise ValueError("non-abstaining candidate answer must be non-empty")
        if self.self_confidence is not None and not 0.0 <= self.self_confidence <= 1.0:
            raise ValueError("self_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class InferenceBundle:
    """Complete production policy input, deliberately without dataset IDs."""

    problem: ProblemInput
    anchor: CandidateEnvelope
    candidates: tuple[CandidateEnvelope, ...] = ()

    @property
    def challengers(self) -> tuple[CandidateEnvelope, ...]:
        """Readable alias for candidates other than the fail-closed anchor."""

        return self.candidates


@dataclass(frozen=True, slots=True)
class FrozenProfile:
    """Pre-declared gates; freeze this before evaluating a target split."""

    name: str = "evidence-os-v1"
    allowed_strong_kinds: frozenset[CertificateKind] = frozenset(
        {
            CertificateKind.EXECUTABLE_CHECK,
            CertificateKind.FORMAL_EQUIVALENCE,
            CertificateKind.CONSTRAINT_SATISFACTION,
            CertificateKind.SOURCE_ENTAILMENT,
            CertificateKind.VISUAL_GROUNDING,
        }
    )
    min_claim_coverage: float = 1.0
    min_deterministic_checks: int = 1
    min_independent_verifiers: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must be non-empty")
        if self.allowed_strong_kinds & WEAK_ONLY_CERTIFICATE_KINDS:
            raise ValueError("agreement and self-confidence cannot be strong evidence")
        if not 0.0 <= self.min_claim_coverage <= 1.0:
            raise ValueError("min_claim_coverage must be between 0 and 1")
        if self.min_deterministic_checks < 1:
            raise ValueError("min_deterministic_checks must be at least 1")
        if self.min_independent_verifiers < 1:
            raise ValueError("min_independent_verifiers must be at least 1")


class DecisionAction(str, Enum):
    KEEP_ANCHOR = "keep_anchor"
    REPLACE_ANCHOR = "replace_anchor"


class DecisionReason(str, Enum):
    NO_CHALLENGERS = "no_challengers"
    EQUIVALENT_TO_ANCHOR = "equivalent_to_anchor"
    INVALID_CHALLENGER = "invalid_challenger"
    NO_STRONG_CERTIFICATE = "no_strong_certificate"
    CONFLICTING_CERTIFICATES = "conflicting_certificates"
    STRONGLY_VERIFIED_CHALLENGER = "strongly_verified_challenger"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: DecisionAction
    selected: CandidateEnvelope
    reason: DecisionReason
    admitted_certificates: tuple[Certificate, ...] = ()


# Backward-readable names for users who think in answers/verifications rather
# than envelopes/certificates.  They are aliases, not additional contracts.
CandidateAnswer = CandidateEnvelope
VerificationCertificate = Certificate
Decision = PolicyDecision
DecisionRequest = InferenceBundle

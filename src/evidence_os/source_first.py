"""Source-first scheduling for the fail-closed Evidence OS policy.

The scheduler is intentionally narrower than the answer-selection policy.  It
may skip an expensive reasoning anchor only when exactly one source answer is
already backed by certificates that the full policy would admit.  If source
evidence is absent, malformed or conflicting, the caller must run the anchor
and then use :mod:`evidence_os.policy` normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .certificates import answer_fingerprint, certificate_fingerprint
from .contracts import (
    CandidateEnvelope,
    Certificate,
    FrozenProfile,
    ProblemInput,
)
from .policy import EvidencePolicy


class SourceFirstAction(str, Enum):
    """Whether an expensive anchor is still needed."""

    RETURN_CERTIFIED_SOURCE = "return_certified_source"
    RUN_ANCHOR = "run_anchor"


class SourceFirstReason(str, Enum):
    """Auditable reason for the scheduling decision."""

    CERTIFIED_SOURCE_IS_DECISIVE = "certified_source_is_decisive"
    NO_CERTIFIED_SOURCE = "no_certified_source"
    CONFLICTING_CERTIFIED_SOURCES = "conflicting_certified_sources"
    ANCHOR_MAY_EMIT_STRONG_CERTIFICATES = "anchor_may_emit_strong_certificates"


@dataclass(frozen=True, slots=True)
class SourceFirstDecision:
    """A scheduling result, not a benchmark verdict."""

    action: SourceFirstAction
    reason: SourceFirstReason
    selected: CandidateEnvelope | None = None
    admitted_certificates: tuple[Certificate, ...] = ()

    @property
    def anchor_required(self) -> bool:
        return self.action is SourceFirstAction.RUN_ANCHOR


def decide_source_first(
    problem: ProblemInput,
    candidates: tuple[CandidateEnvelope, ...],
    *,
    profile: FrozenProfile | None = None,
    anchor_may_emit_strong_certificates: bool = False,
) -> SourceFirstDecision:
    """Decide whether a certified source can safely bypass the anchor.

    The shortcut is answer-equivalent to the full policy under one explicit
    production contract: the deferred anchor is not itself a producer of
    strong certificates.  This matches the current architecture, where the
    reasoning model is an unverified fallback and official source adapters are
    the strong proof producers.  Callers must set
    ``anchor_may_emit_strong_certificates=True`` if that contract changes.
    """

    if anchor_may_emit_strong_certificates:
        return SourceFirstDecision(
            action=SourceFirstAction.RUN_ANCHOR,
            reason=SourceFirstReason.ANCHOR_MAY_EMIT_STRONG_CERTIFICATES,
        )

    active_profile = profile or FrozenProfile()
    policy = EvidencePolicy()
    certified: dict[str, list[tuple[CandidateEnvelope, tuple[Certificate, ...]]]] = {}

    for candidate in candidates:
        if not policy._candidate_is_usable(candidate):
            continue
        admitted = policy.admitted_certificates(candidate, problem, active_profile)
        if len({item.verifier for item in admitted}) < active_profile.min_independent_verifiers:
            continue
        certified.setdefault(answer_fingerprint(candidate), []).append((candidate, admitted))

    if not certified:
        return SourceFirstDecision(
            action=SourceFirstAction.RUN_ANCHOR,
            reason=SourceFirstReason.NO_CERTIFIED_SOURCE,
        )

    if len(certified) != 1:
        conflict = {
            certificate_fingerprint(certificate): certificate
            for group in certified.values()
            for _, certificates in group
            for certificate in certificates
        }
        return SourceFirstDecision(
            action=SourceFirstAction.RUN_ANCHOR,
            reason=SourceFirstReason.CONFLICTING_CERTIFIED_SOURCES,
            admitted_certificates=tuple(conflict[key] for key in sorted(conflict)),
        )

    group = next(iter(certified.values()))
    candidate, _ = min(group, key=lambda item: policy._candidate_sort_key(item[0]))
    admitted_unique = {
        certificate_fingerprint(certificate): certificate
        for _, certificates in group
        for certificate in certificates
    }
    return SourceFirstDecision(
        action=SourceFirstAction.RETURN_CERTIFIED_SOURCE,
        reason=SourceFirstReason.CERTIFIED_SOURCE_IS_DECISIVE,
        selected=candidate,
        admitted_certificates=tuple(
            admitted_unique[key] for key in sorted(admitted_unique)
        ),
    )

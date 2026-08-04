"""Fail-closed and candidate-order-invariant Evidence OS policy."""

from __future__ import annotations

from dataclasses import replace

from .certificates import (
    answer_fingerprint,
    canonicalize_text,
    certificate_fingerprint,
    certificate_matches,
)
from .contracts import (
    CandidateEnvelope,
    Certificate,
    CertificateStrength,
    CertificateVerdict,
    DecisionAction,
    DecisionReason,
    FrozenProfile,
    InferenceBundle,
    PolicyDecision,
)


class EvidencePolicy:
    """Replace the anchor only for one uncontested, strongly proven answer."""

    def decide(self, bundle: InferenceBundle, profile: FrozenProfile) -> PolicyDecision:
        if not bundle.candidates:
            return self._keep(bundle, DecisionReason.NO_CHALLENGERS)

        anchor_key = answer_fingerprint(bundle.anchor)
        member_groups = self._candidate_members(bundle.candidates)
        groups = self._candidate_groups(bundle.candidates)
        challenger_keys = tuple(key for key in groups if key != anchor_key)
        if not challenger_keys:
            return self._keep(bundle, DecisionReason.EQUIVALENT_TO_ANCHOR)

        valid_challenger_keys = tuple(
            key for key in challenger_keys if self._candidate_is_usable(groups[key])
        )
        if not valid_challenger_keys:
            return self._keep(bundle, DecisionReason.INVALID_CHALLENGER)

        admitted_by_answer: dict[str, tuple[Certificate, ...]] = {}
        certified_candidates: dict[str, tuple[CandidateEnvelope, ...]] = {}

        anchor_admitted = self._admitted(bundle.anchor, bundle, profile)
        if len({item.verifier for item in anchor_admitted}) >= profile.min_independent_verifiers:
            admitted_by_answer[anchor_key] = anchor_admitted

        for key in sorted(member_groups):
            admitted_unique: dict[str, Certificate] = {}
            admitted_members: list[CandidateEnvelope] = []
            for candidate in member_groups[key]:
                if not self._candidate_is_usable(candidate):
                    continue
                admitted = self._admitted(candidate, bundle, profile)
                if len({item.verifier for item in admitted}) < profile.min_independent_verifiers:
                    continue
                admitted_members.append(candidate)
                for certificate in admitted:
                    admitted_unique[certificate_fingerprint(certificate)] = certificate
            if admitted_members:
                admitted_by_answer[key] = tuple(
                    admitted_unique[fingerprint]
                    for fingerprint in sorted(admitted_unique)
                )
                certified_candidates[key] = tuple(
                    sorted(admitted_members, key=self._candidate_sort_key)
                )

        certified_challengers = tuple(
            key for key in valid_challenger_keys if key in admitted_by_answer
        )
        if not certified_challengers:
            return self._keep(bundle, DecisionReason.NO_STRONG_CERTIFICATE)

        if len(certified_challengers) != 1 or anchor_key in admitted_by_answer:
            conflict_evidence = tuple(
                certificate
                for key in sorted(admitted_by_answer)
                for certificate in admitted_by_answer[key]
            )
            return self._keep(
                bundle,
                DecisionReason.CONFLICTING_CERTIFICATES,
                conflict_evidence,
            )

        winner_key = certified_challengers[0]
        # Copy the row that actually carried an admitted certificate.  Other
        # branches may emit the same answer text with different reasoning;
        # answer agreement must not transfer a proof to an unrelated row.
        winner = certified_candidates[winner_key][0]
        return PolicyDecision(
            action=DecisionAction.REPLACE_ANCHOR,
            selected=winner,
            reason=DecisionReason.STRONGLY_VERIFIED_CHALLENGER,
            admitted_certificates=admitted_by_answer[winner_key],
        )

    @staticmethod
    def _keep(
        bundle: InferenceBundle,
        reason: DecisionReason,
        admitted: tuple[Certificate, ...] = (),
    ) -> PolicyDecision:
        return PolicyDecision(
            action=DecisionAction.KEEP_ANCHOR,
            selected=bundle.anchor,
            reason=reason,
            admitted_certificates=admitted,
        )

    @staticmethod
    def _candidate_is_usable(candidate: CandidateEnvelope) -> bool:
        return bool(
            candidate.valid_format
            and not candidate.error
            and not candidate.abstain
            and candidate.final_answer.strip()
        )

    @staticmethod
    def _candidate_members(
        candidates: tuple[CandidateEnvelope, ...],
    ) -> dict[str, tuple[CandidateEnvelope, ...]]:
        grouped: dict[str, list[CandidateEnvelope]] = {}
        for candidate in candidates:
            grouped.setdefault(answer_fingerprint(candidate), []).append(candidate)
        return {
            key: tuple(sorted(group, key=EvidencePolicy._candidate_sort_key))
            for key, group in sorted(grouped.items())
        }

    @staticmethod
    def _candidate_groups(
        candidates: tuple[CandidateEnvelope, ...],
    ) -> dict[str, CandidateEnvelope]:
        representatives: dict[str, CandidateEnvelope] = {}
        for key, group in EvidencePolicy._candidate_members(candidates).items():
            usable = [item for item in group if EvidencePolicy._candidate_is_usable(item)]
            representative = min(usable or group, key=EvidencePolicy._candidate_sort_key)
            certificates = {
                certificate_fingerprint(certificate): certificate
                for item in usable
                for certificate in item.certificates
                if isinstance(certificate, Certificate)
            }
            representatives[key] = replace(
                representative,
                certificates=tuple(certificates[item] for item in sorted(certificates)),
            )
        return representatives

    @staticmethod
    def _candidate_sort_key(candidate: CandidateEnvelope) -> tuple[object, ...]:
        return (
            canonicalize_text(candidate.final_answer),
            candidate.final_answer,
            candidate.source,
            candidate.error or "",
            candidate.abstain,
            candidate.valid_format,
            candidate.self_confidence if candidate.self_confidence is not None else -1.0,
        )

    @staticmethod
    def _admitted(
        candidate: CandidateEnvelope,
        bundle: InferenceBundle,
        profile: FrozenProfile,
    ) -> tuple[Certificate, ...]:
        unique: dict[str, Certificate] = {}
        for certificate in candidate.certificates:
            if not isinstance(certificate, Certificate):
                continue
            if certificate.verdict is not CertificateVerdict.PASS:
                continue
            if certificate.strength is not CertificateStrength.STRONG:
                continue
            if certificate.kind not in profile.allowed_strong_kinds:
                continue
            if certificate.claim_coverage < profile.min_claim_coverage:
                continue
            if certificate.contradiction_count:
                continue
            if len(certificate.deterministic_checks) < profile.min_deterministic_checks:
                continue
            if not all(certificate.deterministic_checks):
                continue
            if not certificate.trace_fingerprint:
                continue
            if not certificate_matches(certificate, bundle.problem, candidate):
                continue
            unique[certificate_fingerprint(certificate)] = certificate
        return tuple(unique[key] for key in sorted(unique))


def decide(bundle: InferenceBundle, profile: FrozenProfile | None = None) -> PolicyDecision:
    return EvidencePolicy().decide(bundle, profile or FrozenProfile())


DEFAULT_STRONG_KINDS = FrozenProfile().allowed_strong_kinds
EvidencePolicyConfig = FrozenProfile

"""Adapters from legacy solver rows to typed, fail-closed evidence.

Legacy artifacts contain useful observations, but most of them were not
produced with cryptographic input/answer binding.  This module deliberately
converts those observations to *weak* certificates.  A strong certificate can
only enter through :func:`certificate_from_record`, whose fingerprints must
already match the observable problem and exact candidate answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import AbstractSet, Any

from .certificates import answer_fingerprint, input_fingerprint, issue_certificate
from .contracts import (
    CandidateEnvelope,
    Certificate,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
    ProblemInput,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHOICE = re.compile(r"^\s*([A-E])(?:[).:\s].*)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AdapterAudit:
    """Observable adapter result; it contains no benchmark outcome."""

    source: str
    certificate_observations: tuple[str, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdaptedCandidate:
    candidate: CandidateEnvelope
    audit: AdapterAudit


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _stable_trace(value: Any) -> bytes:
    def jsonable(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): jsonable(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [jsonable(child) for child in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def problem_from_public_payload(
    payload: Mapping[str, Any],
    *,
    image_root: Path | None = None,
) -> tuple[ProblemInput, bool]:
    """Build observable problem content and report whether every image is bound.

    Image paths are never fingerprints: paths commonly contain task IDs.  An
    image contributes to the binding only when its bytes are locally available
    and hashed.  Therefore a visual certificate cannot become strong merely
    because a legacy row mentions ``images/val_0042.png``.
    """

    statement = str(payload.get("question") or payload.get("statement") or "").strip()
    answer_format = str(payload.get("answer_type") or payload.get("answer_format") or "").strip()
    constraints = tuple(
        value
        for value in (
            str(payload.get("subject") or "").strip(),
            str(payload.get("grade") or "").strip(),
        )
        if value
    )

    images = _sequence(payload.get("question_images") or payload.get("images"))
    fingerprints: list[str] = []
    complete = True
    for item in images:
        descriptor = _mapping(item)
        raw_path = descriptor.get("data") or descriptor.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip() or image_root is None:
            complete = False
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            complete = False
            continue
        image_path = (image_root / relative).resolve(strict=False)
        root = image_root.resolve(strict=False)
        try:
            image_path.relative_to(root)
        except ValueError:
            complete = False
            continue
        if not image_path.is_file():
            complete = False
            continue
        fingerprints.append(_sha256_file(image_path))

    if images and len(fingerprints) != len(images):
        complete = False
    if not statement and not fingerprints:
        raise ValueError("public problem payload has neither question text nor bound images")

    return (
        ProblemInput(
            statement=statement,
            image_fingerprints=tuple(sorted(fingerprints)),
            constraints=constraints,
            answer_format=answer_format or None,
        ),
        complete,
    )


def answer_format_valid(answer: str, answer_format: str | None) -> bool:
    if not answer.strip():
        return False
    kind = (answer_format or "").casefold()
    if kind == "choice":
        return bool(_CHOICE.fullmatch(answer))
    if kind == "numeric":
        return bool(re.search(r"[-+]?\d", answer))
    return True


def _bbox_checks(crops: Sequence[Any]) -> tuple[bool, ...]:
    checks: list[bool] = []
    for raw_crop in crops:
        crop = _mapping(raw_crop)
        bbox = _sequence(crop.get("bbox_1000"))
        numeric = len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox)
        ordered = bool(
            numeric
            and 0 <= float(bbox[0]) < float(bbox[2]) <= 1000
            and 0 <= float(bbox[1]) < float(bbox[3]) <= 1000
        )
        checks.append(ordered)
    return tuple(checks)


def adapt_legacy_solver_payload(
    source: str,
    payload: Mapping[str, Any],
    problem: ProblemInput,
    *,
    image_binding_complete: bool,
) -> AdaptedCandidate:
    """Convert a solver payload without trusting model-authored attestations."""

    answer = str(
        payload.get("final_answer")
        or payload.get("prediction")
        or payload.get("answer")
        or ""
    ).strip()
    error_value = payload.get("error")
    error = str(error_value).strip() if error_value not in (None, "") else None
    generation = _mapping(payload.get("generation"))
    certificates: list[Certificate] = []
    observations: list[str] = []
    rejections: list[str] = []

    confidence: float | None = None
    for raw_confidence in (
        generation.get("confidence"),
        _mapping(generation.get("selection_evidence")).get("confidence"),
    ):
        try:
            parsed = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if 0.0 <= parsed <= 1.0:
            confidence = parsed
            break

    citations = tuple(
        str(item).strip()
        for item in _sequence(generation.get("evidence_citations"))
        if str(item).strip()
    )
    if citations:
        observations.append("legacy_rag_citations_present")
        certificates.append(
            issue_certificate(
                problem,
                answer,
                kind=CertificateKind.SOURCE_ENTAILMENT,
                strength=CertificateStrength.WEAK,
                verdict=CertificateVerdict.UNKNOWN,
                verifier="legacy-rag-observation-adapter-v1",
                claim_coverage=0.0,
                contradiction_count=0,
                deterministic_checks=(),
                trace=_stable_trace({"citations": citations}),
            )
        )
        rejections.append("citations_not_roundtrip_or_claim_bound")

    selection_evidence = _mapping(generation.get("selection_evidence"))
    active_crops = _sequence(generation.get("active_crops"))
    if selection_evidence or active_crops:
        observations.append("legacy_visual_observation_present")
        crop_checks = _bbox_checks(active_crops)
        certificates.append(
            issue_certificate(
                problem,
                answer,
                kind=CertificateKind.VISUAL_GROUNDING,
                strength=CertificateStrength.WEAK,
                verdict=CertificateVerdict.UNKNOWN,
                verifier="legacy-active-crop-observation-adapter-v1",
                claim_coverage=0.0,
                contradiction_count=0,
                deterministic_checks=crop_checks,
                trace=_stable_trace(
                    {
                        "crop_checks": crop_checks,
                        "visible_facts": selection_evidence.get("visible_facts", ()),
                    }
                ),
            )
        )
        if not image_binding_complete:
            rejections.append("source_image_bytes_not_bound")
        rejections.append("visual_claims_not_independently_verified")

    calculator = _mapping(generation.get("calculator_sympy"))
    if calculator:
        observations.append("legacy_executable_observation_present")
        program = _mapping(calculator.get("program"))
        draft = _mapping(calculator.get("draft"))
        program_ok = program.get("ok") is True and bool(str(program.get("value") or "").strip())
        independent_answer_matches = (
            str(draft.get("independent_answer") or "").strip().casefold()
            == answer.casefold()
        )
        predicted_value = str(draft.get("predicted_program_value") or "").strip()
        program_value_matches = bool(
            predicted_value
            and predicted_value.casefold() == str(program.get("value") or "").strip().casefold()
        )
        input_bound = bool(calculator.get("input_binding")) and image_binding_complete
        checks = (program_ok, independent_answer_matches, program_value_matches, input_bound)
        verdict = CertificateVerdict.FAIL if not all(checks) else CertificateVerdict.UNKNOWN
        certificates.append(
            issue_certificate(
                problem,
                answer,
                kind=CertificateKind.EXECUTABLE_CHECK,
                strength=CertificateStrength.WEAK,
                verdict=verdict,
                verifier="legacy-calculator-observation-adapter-v1",
                claim_coverage=0.0,
                contradiction_count=0 if all(checks) else 1,
                deterministic_checks=checks,
                trace=_stable_trace(
                    {
                        "program": program,
                        "independent_answer": draft.get("independent_answer"),
                        "predicted_program_value": predicted_value,
                    }
                ),
            )
        )
        if not program_ok:
            rejections.append("program_not_executed")
        if not independent_answer_matches:
            rejections.append("program_answer_not_bound")
        if not program_value_matches:
            rejections.append("program_value_mismatch")
        if not input_bound:
            rejections.append("program_inputs_not_bound_to_observation")

    candidate = CandidateEnvelope(
        source=source,
        final_answer=answer,
        valid_format=answer_format_valid(answer, problem.answer_format),
        error=error,
        abstain=not answer and error is None,
        certificates=tuple(certificates),
        self_confidence=confidence,
    )
    if not candidate.valid_format:
        rejections.append("invalid_answer_format")
    if error:
        rejections.append("solver_error")
    if not certificates:
        rejections.append("no_typed_certificate")

    return AdaptedCandidate(
        candidate=candidate,
        audit=AdapterAudit(
            source=source,
            certificate_observations=tuple(sorted(set(observations))),
            rejection_reasons=tuple(sorted(set(rejections))),
        ),
    )


def certificate_from_record(
    problem: ProblemInput,
    candidate: CandidateEnvelope,
    record: Mapping[str, Any],
    *,
    allowed_verifiers: AbstractSet[str],
    allowed_kinds: AbstractSet[CertificateKind],
    require_inline_trace: bool = True,
) -> Certificate:
    """Load a profile-authorized verifier record after binding checks.

    The caller must first pin the complete certificate artifact in a frozen
    profile.  The verifier and certificate kind are checked against that
    profile here.  An inline trace is required by default, and its digest is
    recomputed instead of trusting the digest declared by the record.
    """

    verifier = str(record.get("verifier") or "")
    if verifier not in allowed_verifiers:
        raise ValueError(f"certificate verifier is not profile-authorized: {verifier!r}")
    kind = CertificateKind(str(record.get("kind") or ""))
    if kind not in allowed_kinds:
        raise ValueError(f"certificate kind is not profile-authorized: {kind.value!r}")

    declared_trace_fingerprint = str(record.get("trace_fingerprint") or "")
    trace = record.get("trace")
    if require_inline_trace:
        if trace is None or (isinstance(trace, str) and not trace.strip()):
            raise ValueError("profile-authorized certificate requires an inline verifier trace")
        trace_bytes = trace.encode("utf-8") if isinstance(trace, str) else _stable_trace(trace)
        actual_trace_fingerprint = hashlib.sha256(trace_bytes).hexdigest()
        if declared_trace_fingerprint != actual_trace_fingerprint:
            raise ValueError("certificate trace fingerprint does not match inline trace")

    certificate = Certificate(
        kind=kind,
        strength=CertificateStrength(str(record.get("strength") or "")),
        verdict=CertificateVerdict(str(record.get("status") or "")),
        input_fingerprint=str(record.get("input_fingerprint") or ""),
        answer_fingerprint=str(record.get("answer_fingerprint") or ""),
        input_bound=record.get("input_bound") is True,
        answer_bound=record.get("answer_bound") is True,
        claim_coverage=float(record.get("claim_coverage", 0.0)),
        contradiction_count=int(record.get("contradiction_count", 0)),
        deterministic_checks=tuple(
            value is True for value in _sequence(record.get("deterministic_checks"))
        ),
        verifier=verifier,
        trace_fingerprint=declared_trace_fingerprint or None,
    )
    if not _HEX64.fullmatch(certificate.input_fingerprint):
        raise ValueError("certificate input fingerprint is not SHA-256")
    if not _HEX64.fullmatch(certificate.answer_fingerprint):
        raise ValueError("certificate answer fingerprint is not SHA-256")
    if certificate.trace_fingerprint and not _HEX64.fullmatch(certificate.trace_fingerprint):
        raise ValueError("certificate trace fingerprint is not SHA-256")
    if certificate.input_fingerprint != input_fingerprint(problem):
        raise ValueError("certificate is bound to a different observable input")
    if certificate.answer_fingerprint != answer_fingerprint(candidate):
        raise ValueError("certificate is bound to a different answer")
    return certificate

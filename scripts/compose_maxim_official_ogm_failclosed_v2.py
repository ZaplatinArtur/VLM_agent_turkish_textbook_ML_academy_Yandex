#!/usr/bin/env python3
"""Overlay independently certified official-source keys on a pinned anchor.

The historical filename is retained for compatibility; the composer accepts
both the OGM API/PDF resolver and reviewed direct-PDF resolver profiles.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.adapters import certificate_from_record  # noqa: E402
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    DecisionAction,
    FrozenProfile,
    InferenceBundle,
    ProblemInput,
)
from evidence_os.official_ogm import (  # noqa: E402
    VERIFIER as OGM_VERIFIER,
    OfficialSourceError,
    observed_source_question_marker,
    parser_observation,
    parser_observation_allow_missing_number,
    parser_observation_primary_layout_number,
    problem_for,
    sha256_file,
)
from evidence_os.official_pdf import VERIFIER as DIRECT_PDF_VERIFIER  # noqa: E402
from evidence_os.official_workbook import (  # noqa: E402
    VERIFIER as WORKBOOK_VERIFIER,
    parse_workbook_index,
    validate_fail_closed_workbook_policy,
)
from evidence_os.policy import EvidencePolicy  # noqa: E402


SCHEMA = "maxim-official-source-failclosed-composition-v2"
_PROFILE_CONTRACTS = {
    "maxim-official-ogm-exact-source-profile-v2": (
        "maxim-official-ogm-exact-source-run-v2",
        OGM_VERIFIER,
        10,
    ),
    "maxim-official-direct-pdf-profile-v2": (
        "maxim-official-direct-pdf-run-v2",
        DIRECT_PDF_VERIFIER,
        5,
    ),
    "maxim-public-workbook-profile-v1": (
        "maxim-public-workbook-run-v1",
        WORKBOOK_VERIFIER,
        8,
    ),
}
_WORKBOOK_TRACE_CHECKS = frozenset(
    {
        "strict_public_document_identity",
        "unique_content_page",
        "unique_source_question_record",
        "question_binding",
        "printed_number_visible_on_page",
        "reviewed_embedded_key",
        "valid_source_answer",
        "source_address_not_task_id",
    }
)


class CompositionError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CompositionError(f"{path}: expected object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompositionError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in output:
            raise CompositionError(f"{label}: missing/duplicate task_id")
        output[task_id] = row
    return output


def _path(configured: str) -> Path:
    raw = Path(configured)
    return raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise CompositionError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _validate_workbook_certificate_artifact(
    *,
    task_id: str,
    raw_candidate: dict[str, Any],
    raw_certificate: dict[str, Any],
    trace: dict[str, Any],
    profile: dict[str, Any],
    profile_sha: str,
) -> None:
    generation = raw_candidate.get("generation")
    if (
        raw_candidate.get("abstain") is not False
        or raw_candidate.get("error") is not None
        or not isinstance(generation, dict)
        or generation.get("gold_access") is not False
        or generation.get("resolver") != WORKBOOK_VERIFIER
        or generation.get("source_certificate") is not True
    ):
        raise CompositionError(f"workbook candidate {task_id} violates resolver invariants")
    if (
        raw_certificate.get("schema_version")
        != "maxim-public-workbook-certificate-v1"
        or raw_certificate.get("verifier") != WORKBOOK_VERIFIER
        or trace.get("schema_version") != "public-workbook-source-trace-v1"
        or trace.get("verifier") != WORKBOOK_VERIFIER
        or trace.get("accepted") is not True
    ):
        raise CompositionError(f"workbook certificate {task_id} has an invalid contract")
    trace_checks = trace.get("checks")
    deterministic_checks = raw_certificate.get("deterministic_checks")
    if (
        not isinstance(trace_checks, dict)
        or set(trace_checks) != _WORKBOOK_TRACE_CHECKS
        or any(value is not True for value in trace_checks.values())
        or not isinstance(deterministic_checks, list)
        or len(deterministic_checks) != len(_WORKBOOK_TRACE_CHECKS)
        or any(value is not True for value in deterministic_checks)
    ):
        raise CompositionError(f"workbook certificate {task_id} has inconsistent checks")
    inputs = profile.get("inputs")
    runtime = profile.get("runtime")
    documents = profile.get("documents")
    provenance = trace.get("provenance")
    trace_source = trace.get("source")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (inputs, dict),
            (runtime, dict),
            (documents, list),
            (provenance, dict),
            (trace_source, dict),
        )
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks frozen provenance")
    document_id = str(trace_source.get("document_id") or "")
    frozen_documents = {
        str(item.get("document_id") or ""): item
        for item in documents
        if isinstance(item, dict)
    }
    frozen_document = frozen_documents.get(document_id)
    if frozen_document is None:
        raise CompositionError(f"workbook certificate {task_id} names an unknown document")
    expected_provenance = {
        "profile_sha256": profile_sha,
        "parser_observations_sha256": str(inputs["parser_observations"]["sha256"]),
        "source_locators_sha256": str(inputs["source_locators"]["sha256"]),
        "source_index_sha256": str(inputs["source_index"]["sha256"]),
        "workbook_pdf_sha256": str(frozen_document["pdf_sha256"]),
        "pypdf_version": str(runtime["pypdf_version"]),
        "pdfplumber_version": str(runtime["pdfplumber_version"]),
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise CompositionError(f"workbook certificate {task_id} provenance was altered")
    source_verification = provenance.get("source_verification")
    content_marker_counts = (
        source_verification.get("content_marker_counts")
        if isinstance(source_verification, dict)
        else None
    )
    answer_format = trace_source.get("answer_format")
    key_binding_kind = trace_source.get("key_binding_kind")
    projection_sha = str(trace_source.get("key_projection_sha256") or "")
    content_projection_sha = str(
        trace_source.get("content_projection_sha256") or ""
    )
    question_marker_kind = str(
        trace_source.get("question_marker_kind") or "numbered_item"
    )
    key_context_page_number = trace_source.get("key_context_page_number")
    binding_is_supported = (
        answer_format == "choice"
        and key_binding_kind in {
            "inline_solution",
            "answer_key_table",
            "answer_key_list",
        }
        and not projection_sha
        and not content_projection_sha
        and question_marker_kind == "numbered_item"
    ) or (
        answer_format == "short_text"
        and key_binding_kind == "coordinate_answer_key"
        and len(projection_sha) == 64
        and all(character in "0123456789abcdef" for character in projection_sha)
        and not content_projection_sha
        and question_marker_kind == "numbered_item"
    ) or (
        answer_format == "short_text"
        and key_binding_kind == "coordinate_table_answer_key"
        and question_marker_kind in {"numbered_item", "example_label"}
        and isinstance(key_context_page_number, int)
        and not isinstance(key_context_page_number, bool)
        and key_context_page_number >= 1
        and len(projection_sha) == 64
        and all(character in "0123456789abcdef" for character in projection_sha)
        and len(content_projection_sha) == 64
        and all(
            character in "0123456789abcdef"
            for character in content_projection_sha
        )
    )
    if (
        trace_source.get("pdf_sha256") != frozen_document.get("pdf_sha256")
        or not binding_is_supported
        or not isinstance(source_verification, dict)
        or int(source_verification.get("records", 0)) < 1
        or source_verification.get("verified_records")
        != source_verification.get("records")
        or not isinstance(content_marker_counts, dict)
        or content_marker_counts.get(str(trace_source.get("record_id") or "")) != 1
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks PDF-bound evidence")


def _validate_workbook_trace_against_source(
    *,
    task_id: str,
    candidate_answer: str,
    observation: Any,
    trace: dict[str, Any],
    source_records: dict[str, tuple[Any, Any]],
    allow_example_label_marker: bool,
) -> None:
    trace_source = trace.get("source")
    trace_observation = trace.get("observation")
    trace_match = trace.get("match")
    if not all(
        isinstance(value, dict)
        for value in (trace_source, trace_observation, trace_match)
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks source trace")
    record_id = str(trace_source.get("record_id") or "")
    source_entry = source_records.get(record_id)
    if source_entry is None:
        raise CompositionError(
            f"workbook certificate {task_id} does not name a frozen source record"
        )
    document, question = source_entry
    expected_content_bbox = (
        list(question.content_bbox) if question.content_bbox is not None else None
    )
    expected_source = {
        "document_id": document.document_id,
        "public_locator": document.identity.public_locator,
        "name": document.identity.name,
        "pdf_sha256": document.pdf_sha256,
        "matched_page_number": question.content_page_number,
        "question_number": question.question_number,
        "question_marker_kind": question.question_marker_kind,
        "record_id": question.record_id,
        "answer_format": question.answer_format,
        "key_binding_kind": question.key_binding_kind,
        "key_page_number": question.key_page_number,
        "key_context_page_number": question.key_context_page_number,
        "key_bbox": list(question.key_bbox),
        "content_bbox": expected_content_bbox,
        "key_projection_sha256": question.key_projection_sha256,
        "content_projection_sha256": question.content_projection_sha256,
    }
    actual_source = {
        key: trace_source.get(key) for key in expected_source
    }
    if question.key_binding_kind != "coordinate_table_answer_key":
        actual_source["question_marker_kind"] = str(
            trace_source.get("question_marker_kind") or "numbered_item"
        )
        actual_source["key_context_page_number"] = trace_source.get(
            "key_context_page_number", question.key_page_number
        )
        actual_source["key_projection_sha256"] = str(
            trace_source.get("key_projection_sha256") or ""
        )
        actual_source["content_projection_sha256"] = str(
            trace_source.get("content_projection_sha256") or ""
        )
    if any(actual_source[key] != value for key, value in expected_source.items()):
        raise CompositionError(
            f"workbook certificate {task_id} source trace differs from frozen index"
        )
    if candidate_answer != question.answer.strip():
        raise CompositionError(
            f"workbook certificate {task_id} answer differs from frozen source"
        )
    marker_kind, marker_number = observed_source_question_marker(observation)
    if (
        marker_kind != question.question_marker_kind
        or marker_number != question.question_number
        or (
            marker_kind == "example_label"
            and not allow_example_label_marker
        )
    ):
        raise CompositionError(
            f"workbook certificate {task_id} lacks one matching observed source marker"
        )
    expected_observation = {
        "image_sha256": observation.image_sha256,
        "image_size": [observation.width, observation.height],
        "observed_question_number": observation.question_number,
        "observed_source_marker_kind": marker_kind,
        "observed_source_marker_number": marker_number,
        "parser_identity": observation.parser_identity,
    }
    actual_observation = {
        key: trace_observation.get(key) for key in expected_observation
    }
    if (
        question.key_binding_kind != "coordinate_table_answer_key"
        and question.question_marker_kind == "numbered_item"
    ):
        actual_observation["observed_source_marker_kind"] = str(
            trace_observation.get("observed_source_marker_kind") or "numbered_item"
        )
        actual_observation["observed_source_marker_number"] = trace_observation.get(
            "observed_source_marker_number", observation.question_number
        )
    if any(
        actual_observation[key] != value
        for key, value in expected_observation.items()
    ):
        raise CompositionError(
            f"workbook certificate {task_id} observation trace was altered"
        )
    expected_binding_method = (
        "source_visible_example_label"
        if marker_kind == "example_label"
        else "printed_number"
    )
    if trace_match.get("question_binding_method") != expected_binding_method:
        raise CompositionError(
            f"workbook certificate {task_id} binding method conflicts with its marker"
        )


def compose(profile_path: Path, resolver_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    profile = _load_json(profile_path)
    profile_schema = str(profile.get("schema_version") or "")
    contract = _PROFILE_CONTRACTS.get(profile_schema)
    if contract is None:
        raise CompositionError("unsupported profile schema")
    resolver_schema, verifier, min_checks = contract
    if profile_schema == "maxim-public-workbook-profile-v1":
        policy = profile.get("policy")
        if not isinstance(policy, dict):
            raise CompositionError("workbook profile policy is missing")
        number_projection, allow_example_label_marker = (
            validate_fail_closed_workbook_policy(policy)
        )
        if number_projection == "unique_block_markers_v1":
            observation_loader = parser_observation_allow_missing_number
        elif number_projection == "primary_layout_then_unique_v1":
            observation_loader = parser_observation_primary_layout_number
        else:
            raise CompositionError("unsupported workbook question-number projection")
    else:
        observation_loader = parser_observation
        allow_example_label_marker = False
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile.get("expected_rows", 0))
    anchor_spec = profile.get("anchor")
    inputs = profile.get("inputs")
    if not isinstance(anchor_spec, dict) or not isinstance(inputs, dict):
        raise CompositionError("profile anchor and inputs are required")
    anchor_path = _path(str(anchor_spec.get("path") or ""))
    parser_path = _path(str(inputs["parser_observations"]["path"]))
    locator_path = _path(str(inputs["source_locators"]["path"]))
    _require_hash(anchor_path, str(anchor_spec.get("sha256") or ""), "anchor")
    _require_hash(
        parser_path, str(inputs["parser_observations"].get("sha256") or ""), "parser"
    )
    _require_hash(
        locator_path, str(inputs["source_locators"].get("sha256") or ""), "source locators"
    )
    if profile_schema == "maxim-public-workbook-profile-v1":
        source_index_path = _path(str(inputs["source_index"]["path"]))
        _require_hash(
            source_index_path,
            str(inputs["source_index"].get("sha256") or ""),
            "workbook source index",
        )
        workbook_index = parse_workbook_index(_load_json(source_index_path))
        workbook_source_records = {
            question.record_id: (document, question)
            for document in workbook_index.documents
            for question in document.questions
        }
    else:
        workbook_source_records = {}
    resolver_manifest = _load_json(resolver_manifest_path)
    if resolver_manifest.get("schema_version") != resolver_schema:
        raise CompositionError("resolver manifest schema is invalid")
    if resolver_manifest.get("gold_access") is not False or resolver_manifest.get(
        "benchmark_candidate_or_outcome_access"
    ) is not False:
        raise CompositionError("resolver manifest lacks strict blind attestations")
    manifest_profile = resolver_manifest.get("profile")
    artifacts = resolver_manifest.get("artifacts")
    if not isinstance(manifest_profile, dict) or not isinstance(artifacts, dict):
        raise CompositionError("resolver manifest is incomplete")
    if manifest_profile.get("sha256") != profile_sha:
        raise CompositionError("resolver was produced under a different frozen profile")
    if profile_schema == "maxim-public-workbook-profile-v1":
        manifest_inputs = resolver_manifest.get("inputs")
        if not isinstance(manifest_inputs, dict):
            raise CompositionError("workbook resolver manifest has no pinned inputs")
        for key in ("parser_observations", "source_locators", "source_index"):
            manifest_item = manifest_inputs.get(key)
            if (
                not isinstance(manifest_item, dict)
                or manifest_item.get("sha256") != inputs[key].get("sha256")
            ):
                raise CompositionError(f"workbook resolver {key} provenance changed")
        manifest_documents = manifest_inputs.get("documents")
        frozen_documents = profile.get("documents")
        if not isinstance(manifest_documents, dict) or not isinstance(frozen_documents, list):
            raise CompositionError("workbook resolver document provenance is incomplete")
        frozen_document_hashes = {
            str(item.get("document_id") or ""): str(item.get("pdf_sha256") or "")
            for item in frozen_documents
            if isinstance(item, dict)
        }
        if set(manifest_documents) != set(frozen_document_hashes) or any(
            not isinstance(manifest_documents[document_id], dict)
            or manifest_documents[document_id].get("sha256") != expected_hash
            for document_id, expected_hash in frozen_document_hashes.items()
        ):
            raise CompositionError("workbook resolver PDF provenance changed")
    candidate_path = Path(str(artifacts["candidate"]["path"])).resolve()
    certificate_path = Path(str(artifacts["certificates"]["path"])).resolve()
    _require_hash(candidate_path, str(artifacts["candidate"]["sha256"]), "resolver candidate")
    _require_hash(
        certificate_path,
        str(artifacts["certificates"]["sha256"]),
        "resolver certificates",
    )

    anchor_rows = _load_jsonl(anchor_path)
    parser_rows = _load_jsonl(parser_path)
    locator_rows = _load_jsonl(locator_path)
    candidate_rows = _load_jsonl(candidate_path)
    certificate_rows = _load_jsonl(certificate_path)
    if len(anchor_rows) != expected_rows or len(parser_rows) != expected_rows:
        raise CompositionError(f"anchor/parser must contain exactly {expected_rows} rows")
    anchor = _index(anchor_rows, "anchor")
    parser = _index(parser_rows, "parser")
    locators = _index(locator_rows, "source locators")
    candidates = _index(candidate_rows, "resolver candidate")
    certificates = _index(certificate_rows, "resolver certificates")
    if profile_schema == "maxim-public-workbook-profile-v1" and (
        resolver_manifest.get("accepted_certificates") != len(certificates)
        or resolver_manifest.get("rows") != expected_rows
        or resolver_manifest.get("task_id_used_for_alignment_only") is not True
        or resolver_manifest.get("viewer_nosw_used_as_policy_feature") is not False
    ):
        raise CompositionError("workbook resolver manifest counters or invariants changed")
    task_set = set(anchor)
    if set(parser) != task_set or set(candidates) != task_set or not task_set <= set(locators):
        raise CompositionError("resolver, parser, locator, and anchor task sets do not align")
    if not set(certificates) <= task_set:
        raise CompositionError("certificate artifact contains an unknown task")

    evidence_policy = EvidencePolicy()
    frozen_policy = FrozenProfile(
        name=str(profile.get("profile_name") or "maxim-official-ogm-v2"),
        allowed_strong_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
        min_claim_coverage=1.0,
        min_deterministic_checks=min_checks,
        min_independent_verifiers=1,
    )
    output_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    overrides = 0
    certified_equal_anchor = 0
    for raw_anchor in anchor_rows:
        task_id = str(raw_anchor["task_id"])
        anchor_answer = str(raw_anchor.get("final_answer") or "").strip()
        if not anchor_answer:
            raise CompositionError(f"anchor {task_id} has no final_answer")
        anchor_candidate = CandidateEnvelope(source="fail-closed-anchor", final_answer=anchor_answer)
        raw_candidate = candidates[task_id]
        candidate_answer = str(raw_candidate.get("final_answer") or "").strip()
        challenger_tuple: tuple[CandidateEnvelope, ...] = ()
        certificate_fingerprint = None
        if task_id in certificates:
            if not candidate_answer:
                raise CompositionError(f"certified candidate {task_id} has no answer")
            generation = raw_candidate.get("generation")
            if not isinstance(generation, dict) or generation.get("gold_access") is not False:
                raise CompositionError(f"candidate {task_id} lacks a strict blind attestation")
            observation = observation_loader(parser[task_id])
            source_url = str(locators[task_id].get("source_url") or "")
            trace = certificates[task_id].get("trace")
            if not isinstance(trace, dict) or trace.get("accepted") is not True:
                raise CompositionError(f"certificate {task_id} trace is not accepted")
            if profile_schema == "maxim-public-workbook-profile-v1":
                _validate_workbook_certificate_artifact(
                    task_id=task_id,
                    raw_candidate=raw_candidate,
                    raw_certificate=certificates[task_id],
                    trace=trace,
                    profile=profile,
                    profile_sha=profile_sha,
                )
                _validate_workbook_trace_against_source(
                    task_id=task_id,
                    candidate_answer=candidate_answer,
                    observation=observation,
                    trace=trace,
                    source_records=workbook_source_records,
                    allow_example_label_marker=allow_example_label_marker,
                )
                trace_source = trace.get("source")
                answer_format = (
                    str(trace_source.get("answer_format") or "")
                    if isinstance(trace_source, dict)
                    else ""
                )
                if answer_format not in {"choice", "short_text"}:
                    raise CompositionError(
                        f"certificate {task_id} has no valid source answer format"
                    )
            else:
                answer_format = "choice"
            problem = problem_for(
                observation,
                source_url,
                answer_format=answer_format,
            )
            bare = CandidateEnvelope(source=verifier, final_answer=candidate_answer)
            certificate = certificate_from_record(
                problem,
                bare,
                certificates[task_id],
                allowed_verifiers=frozenset({verifier}),
                allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
                require_inline_trace=True,
            )
            provenance = trace.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("profile_sha256") != profile_sha:
                raise CompositionError(f"certificate {task_id} is not profile-bound")
            challenger_tuple = (
                CandidateEnvelope(
                    source=verifier,
                    final_answer=candidate_answer,
                    certificates=(certificate,),
                ),
            )
            certificate_fingerprint = certificate.trace_fingerprint
            bundle = InferenceBundle(
                problem=problem,
                anchor=anchor_candidate,
                candidates=challenger_tuple,
            )
            decision = evidence_policy.decide(bundle, frozen_policy)
        else:
            # No certificate means no policy feature is consulted.  Avoid even
            # parsing OCR for an ineligible row; it cannot affect the anchor.
            problem = ProblemInput(statement="No certified challenger was admitted.")
            decision = evidence_policy.decide(
                InferenceBundle(problem=problem, anchor=anchor_candidate), frozen_policy
            )
        reasons[decision.reason.value] += 1
        output = dict(raw_anchor)
        if decision.action is DecisionAction.REPLACE_ANCHOR:
            overrides += 1
            output["final_answer"] = decision.selected.final_answer
            generation = output.get("generation")
            copied_generation = dict(generation) if isinstance(generation, dict) else {}
            copied_generation["official_source_override"] = {
                "verifier": verifier,
                "trace_fingerprint": certificate_fingerprint,
                "profile_sha256": profile_sha,
            }
            output["generation"] = copied_generation
        elif challenger_tuple and candidate_answer == anchor_answer:
            certified_equal_anchor += 1
        output_rows.append(output)
        decisions.append(
            {
                "task_id": task_id,
                "action": decision.action.value,
                "reason": decision.reason.value,
                "anchor_answer": anchor_answer,
                "selected_answer": str(output["final_answer"]),
                "certificate_trace_fingerprint": certificate_fingerprint,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    solver_path = output_dir / "solver.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    _write_jsonl(solver_path, output_rows)
    _write_jsonl(decisions_path, decisions)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": expected_rows,
        "overrides": overrides,
        "certified_equal_anchor": certified_equal_anchor,
        "certificates_seen": len(certificates),
        "decision_reasons": dict(sorted(reasons.items())),
        "gold_access": False,
        "score_or_outcome_access": False,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "resolver_manifest": {
            "path": str(resolver_manifest_path),
            "sha256": sha256_file(resolver_manifest_path),
        },
        "anchor": {"path": str(anchor_path), "sha256": sha256_file(anchor_path)},
        "output": {
            "solver": {"path": str(solver_path), "sha256": sha256_file(solver_path)},
            "decisions": {"path": str(decisions_path), "sha256": sha256_file(decisions_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compose(
            args.profile_json.resolve(),
            args.resolver_manifest.resolve(),
            args.output_dir.resolve(),
        )
    except (
        CompositionError,
        OfficialSourceError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

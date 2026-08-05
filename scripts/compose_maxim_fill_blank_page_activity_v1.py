#!/usr/bin/env python3
"""Overlay replayed fill-blank certificates on an opaque pinned V6 anchor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.fill_blank_page_activity import (  # noqa: E402
    TRACE_SCHEMA,
    VERIFIER,
    FillBlankPageThresholds,
    document_for_source,
    parse_fill_blank_page_index,
    resolve_fill_blank_page_activity,
    verify_fill_blank_page_index_pdf,
)
from evidence_os.official_ogm import (  # noqa: E402
    OfficialSourceError,
    PageMatcher,
    canonical_json_bytes,
    parser_observation_primary_layout_number,
    sha256_file,
)
from scripts.run_maxim_fill_blank_page_activity_v1 import (  # noqa: E402
    CERTIFICATE_SCHEMA,
    PROFILE_SCHEMA,
    RUN_SCHEMA,
    _certificate_record,
    _validate_profile,
)


COMPOSITION_SCHEMA = "maxim-fill-blank-page-activity-composition-v1"
_EXPECTED_CHECKS = frozenset(
    {
        "strict_public_document_identity",
        "unique_content_page",
        "no_single_source_question_marker",
        "unique_page_activity_record",
        "source_activity_text_match",
        "complete_numbered_item_inventory",
        "pdf_activity_title_attested",
        "pdf_activity_instruction_attested",
        "pdf_complete_item_inventory_attested",
        "pdf_word_bank_multiset_attested",
        "pdf_answer_key_components_attested",
        "pdf_attestation_replayed",
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
            raise CompositionError(f"{label}: missing/duplicate alignment key")
        output[task_id] = row
    return output


def _repo_path(configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise CompositionError(f"{label} SHA-256 mismatch")
    return actual


def _load_anchor_rows(
    path: Path,
) -> list[tuple[bytes, dict[str, Any]]]:
    """Read only alignment keys; all unchanged anchor bytes remain opaque."""

    output: list[tuple[bytes, dict[str, Any]]] = []
    seen: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CompositionError(
                    f"opaque anchor line {line_number} is malformed"
                ) from exc
            if not isinstance(value, dict):
                raise CompositionError("opaque anchor row is not an object")
            task_id = str(value.get("task_id") or "").strip()
            if not task_id or task_id in seen:
                raise CompositionError("opaque anchor alignment keys are malformed")
            seen.add(task_id)
            output.append((raw_line.rstrip(b"\r\n"), value))
    return output


def _validate_resolver_contract(
    profile: dict[str, Any],
    profile_sha: str,
    resolver_manifest: dict[str, Any],
) -> None:
    if (
        resolver_manifest.get("schema_version") != RUN_SCHEMA
        or resolver_manifest.get("gold_access") is not False
        or resolver_manifest.get("benchmark_candidate_or_outcome_access") is not False
        or resolver_manifest.get("task_id_used_for_alignment_only") is not True
        or resolver_manifest.get("rows") != profile.get("expected_rows")
    ):
        raise CompositionError("fill-blank resolver manifest contract changed")
    manifest_profile = resolver_manifest.get("profile")
    artifacts = resolver_manifest.get("artifacts")
    inputs = resolver_manifest.get("inputs")
    if not all(isinstance(value, dict) for value in (manifest_profile, artifacts, inputs)):
        raise CompositionError("fill-blank resolver manifest is incomplete")
    if manifest_profile.get("sha256") != profile_sha:
        raise CompositionError("fill-blank resolver profile hash changed")
    for key in ("parser_observations", "source_locators", "source_index"):
        if inputs.get(key, {}).get("sha256") != profile["inputs"][key]["sha256"]:
            raise CompositionError(f"fill-blank resolver {key} provenance changed")


def compose(
    profile_path: Path,
    resolver_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    _validate_profile(profile)
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise CompositionError("unsupported fill-blank profile")
    profile_sha = sha256_file(profile_path)
    resolver_manifest = _load_json(resolver_manifest_path)
    _validate_resolver_contract(profile, profile_sha, resolver_manifest)
    expected_rows = int(profile["expected_rows"])
    inputs = profile["inputs"]
    parser_path = _repo_path(str(inputs["parser_observations"]["path"]))
    locator_path = _repo_path(str(inputs["source_locators"]["path"]))
    source_index_path = _repo_path(str(inputs["source_index"]["path"]))
    _require_hash(parser_path, str(inputs["parser_observations"]["sha256"]), "parser")
    _require_hash(locator_path, str(inputs["source_locators"]["sha256"]), "locators")
    _require_hash(source_index_path, str(inputs["source_index"]["sha256"]), "source index")
    parser_rows = _load_jsonl(parser_path)
    locator_rows = _load_jsonl(locator_path)
    if len(parser_rows) != expected_rows:
        raise CompositionError("fill-blank parser row count changed")
    parser = _index(parser_rows, "parser")
    locators = _index(locator_rows, "locators")
    if not set(parser) <= set(locators):
        raise CompositionError("fill-blank source locators do not cover parser")
    artifacts = resolver_manifest["artifacts"]
    candidate_path = Path(str(artifacts["candidate"]["path"])).resolve()
    certificate_path = Path(str(artifacts["certificates"]["path"])).resolve()
    _require_hash(candidate_path, str(artifacts["candidate"]["sha256"]), "candidate")
    _require_hash(
        certificate_path,
        str(artifacts["certificates"]["sha256"]),
        "certificates",
    )
    candidates = _index(_load_jsonl(candidate_path), "candidate")
    certificates = _index(_load_jsonl(certificate_path), "certificate")
    if set(candidates) != set(parser) or not set(certificates) <= set(parser):
        raise CompositionError("fill-blank resolver alignment keys changed")
    if resolver_manifest.get("accepted_certificates") != len(certificates):
        raise CompositionError("fill-blank certificate counter changed")
    index = parse_fill_blank_page_index(_load_json(source_index_path))
    documents = {document.document_id: document for document in index.documents}
    records = {
        activity.record_id: (document, activity)
        for document in index.documents
        for activity in document.activities
    }
    try:
        import pdfplumber
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise CompositionError("fill-blank composition requires PDF runtime") from exc
    runtime = profile["runtime"]
    if (
        str(pdfplumber.__version__) != str(runtime["pdfplumber_version"])
        or str(pypdf.__version__) != str(runtime["pypdf_version"])
    ):
        raise CompositionError("fill-blank composition PDF runtime changed")
    manifest_documents = resolver_manifest["inputs"].get("documents")
    if not isinstance(manifest_documents, dict) or set(manifest_documents) != set(documents):
        raise CompositionError("fill-blank resolver PDF provenance is incomplete")
    caches: dict[str, dict[str, Any]] = {}
    for document_id, document in documents.items():
        raw = manifest_documents[document_id]
        if not isinstance(raw, dict) or raw.get("sha256") != document.pdf_sha256:
            raise CompositionError("fill-blank resolver PDF hash changed")
        pdf_path = Path(str(raw.get("path") or "")).resolve()
        _require_hash(pdf_path, document.pdf_sha256, "fill-blank PDF")
        reader = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if len(page_texts) != document.page_count:
            raise CompositionError("fill-blank PDF page count changed")
        caches[document_id] = {
            "path": pdf_path,
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
            "source_verification": verify_fill_blank_page_index_pdf(
                pdf_path, document
            ),
        }
    policy = profile["policy"]
    thresholds = FillBlankPageThresholds(
        min_page_coverage=float(policy["min_page_coverage"]),
        min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
        min_page_margin=float(policy["min_page_margin"]),
        min_activity_coverage=float(policy["min_activity_coverage"]),
        min_activity_matched_tokens=int(policy["min_activity_matched_tokens"]),
    )
    verified_answers: dict[str, str] = {}
    trace_fingerprints: dict[str, str] = {}
    for task_id, raw_certificate in certificates.items():
        raw_candidate = candidates[task_id]
        generation = raw_candidate.get("generation")
        trace = raw_certificate.get("trace")
        if (
            raw_candidate.get("abstain") is not False
            or raw_candidate.get("error") is not None
            or not isinstance(generation, dict)
            or generation.get("gold_access") is not False
            or generation.get("resolver") != VERIFIER
            or generation.get("source_certificate") is not True
            or raw_certificate.get("schema_version") != CERTIFICATE_SCHEMA
            or raw_certificate.get("verifier") != VERIFIER
            or raw_certificate.get("kind") != "source_entailment"
            or raw_certificate.get("strength") != "strong"
            or raw_certificate.get("status") != "pass"
            or not isinstance(trace, dict)
            or trace.get("schema_version") != TRACE_SCHEMA
            or trace.get("verifier") != VERIFIER
            or trace.get("accepted") is not True
        ):
            raise CompositionError(f"fill-blank certificate {task_id} is malformed")
        trace_checks = trace.get("checks")
        deterministic_checks = raw_certificate.get("deterministic_checks")
        if (
            not isinstance(trace_checks, dict)
            or set(trace_checks) != _EXPECTED_CHECKS
            or any(value is not True for value in trace_checks.values())
            or not isinstance(deterministic_checks, list)
            or len(deterministic_checks) != len(_EXPECTED_CHECKS)
            or any(value is not True for value in deterministic_checks)
        ):
            raise CompositionError(f"fill-blank checks {task_id} were altered")
        trace_source = trace.get("source")
        provenance = trace.get("provenance")
        if not isinstance(trace_source, dict) or not isinstance(provenance, dict):
            raise CompositionError(f"fill-blank trace {task_id} lacks provenance")
        record_id = str(trace_source.get("record_id") or "")
        source_entry = records.get(record_id)
        if source_entry is None:
            raise CompositionError(f"fill-blank trace {task_id} names unknown source")
        document, activity = source_entry
        cache = caches[document.document_id]
        expected_source = {
            "document_id": document.document_id,
            "public_locator": document.identity.public_locator,
            "name": document.identity.name,
            "pdf_sha256": document.pdf_sha256,
            "matched_page_number": activity.content_page_number,
            "record_id": activity.record_id,
            "answer_format": "short_text",
            "question_marker_kind": "numberless_page_activity",
            "key_binding_kind": "fill_blank_answer_key",
            "key_page_number": activity.key_page_number,
            "key_context_page_number": activity.key_context_page_number,
            "content_bbox": list(activity.content_bbox),
            "word_bank_bbox": list(activity.word_bank_bbox),
            "key_bbox": list(activity.key_bbox),
            "expected_item_count": activity.expected_item_count,
            "expected_column_count": activity.expected_column_count,
            "key_projection_sha256": activity.key_projection_sha256,
            "content_projection_sha256": activity.content_projection_sha256,
            "binding_projection_sha256": activity.binding_projection_sha256,
        }
        if any(trace_source.get(key) != value for key, value in expected_source.items()):
            raise CompositionError(f"fill-blank trace {task_id} differs from source")
        candidate_answer = str(raw_candidate.get("final_answer") or "")
        if candidate_answer != activity.answer:
            raise CompositionError(f"fill-blank candidate {task_id} differs from source")
        expected_provenance = {
            "profile_sha256": profile_sha,
            "parser_observations_sha256": inputs["parser_observations"]["sha256"],
            "source_locators_sha256": inputs["source_locators"]["sha256"],
            "source_index_sha256": inputs["source_index"]["sha256"],
            "fill_blank_pdf_sha256": document.pdf_sha256,
            "pypdf_version": str(pypdf.__version__),
            "pdfplumber_version": str(pdfplumber.__version__),
            "source_verification": cache["source_verification"],
        }
        if provenance != expected_provenance:
            raise CompositionError(f"fill-blank provenance {task_id} was altered")
        observation = parser_observation_primary_layout_number(parser[task_id])
        source_url = str(locators[task_id].get("source_url") or "")
        if document_for_source(index, source_url, allow_missing_nosw=True) != document:
            raise CompositionError(f"fill-blank source identity {task_id} changed")
        replay = resolve_fill_blank_page_activity(
            observation,
            source_url,
            document,
            cache["matcher"],
            cache["page_texts"],
            thresholds,
            verified_record_attestations=cache["source_verification"][
                "record_attestations"
            ],
            allow_missing_nosw=True,
        )
        replay.trace["provenance"] = expected_provenance
        if (
            not replay.accepted
            or replay.answer != candidate_answer
            or replay.trace != trace
        ):
            raise CompositionError(f"fill-blank certificate {task_id} cannot replay")
        expected_certificate = _certificate_record(task_id, replay, candidate_answer)
        if raw_certificate != expected_certificate:
            raise CompositionError(f"fill-blank certificate {task_id} fingerprint changed")
        verified_answers[task_id] = candidate_answer
        trace_fingerprints[task_id] = str(raw_certificate["trace_fingerprint"])
    anchor_spec = profile["anchor"]
    anchor_path = _repo_path(str(anchor_spec["path"]))
    anchor_sha = _require_hash(anchor_path, str(anchor_spec["sha256"]), "opaque V6 anchor")
    anchor_rows = _load_anchor_rows(anchor_path)
    if len(anchor_rows) != expected_rows:
        raise CompositionError("opaque V6 anchor row count changed")
    anchor_ids = {str(row["task_id"]) for _raw, row in anchor_rows}
    if anchor_ids != set(parser) or not set(verified_answers) <= anchor_ids:
        raise CompositionError("opaque anchor alignment keys differ from parser")
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_path = output_dir / "solver.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    temporary = solver_path.with_suffix(".jsonl.tmp")
    decisions: list[dict[str, Any]] = []
    with temporary.open("wb") as sink:
        for raw_line, anchor_row in anchor_rows:
            task_id = str(anchor_row["task_id"])
            if task_id not in verified_answers:
                sink.write(raw_line + b"\n")
                decisions.append(
                    {
                        "task_id": task_id,
                        "source_override": False,
                        "anchor_bytes_copied": True,
                    }
                )
                continue
            output = dict(anchor_row)
            output["final_answer"] = verified_answers[task_id]
            generation = output.get("generation")
            copied_generation = dict(generation) if isinstance(generation, dict) else {}
            copied_generation["fill_blank_page_activity_override"] = {
                "verifier": VERIFIER,
                "trace_fingerprint": trace_fingerprints[task_id],
                "anchor_answer_compared": False,
            }
            output["generation"] = copied_generation
            sink.write(canonical_json_bytes(output) + b"\n")
            decisions.append(
                {
                    "task_id": task_id,
                    "source_override": True,
                    "anchor_bytes_copied": False,
                    "certificate_trace_fingerprint": trace_fingerprints[task_id],
                }
            )
    temporary.replace(solver_path)
    with decisions_path.open("wb") as sink:
        for decision in decisions:
            sink.write(canonical_json_bytes(decision) + b"\n")
    manifest = {
        "schema_version": COMPOSITION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "task_id_used_for_alignment_only": True,
        "anchor_answer_used_as_policy_feature": False,
        "anchor_answer_compared": False,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "resolver_manifest": {
            "path": str(resolver_manifest_path),
            "sha256": sha256_file(resolver_manifest_path),
        },
        "anchor": {"path": str(anchor_path), "sha256": anchor_sha},
        "rows": expected_rows,
        "source_overrides": len(verified_answers),
        "opaque_anchor_rows_copied": expected_rows - len(verified_answers),
        "artifacts": {
            "solver": {"path": str(solver_path), "sha256": sha256_file(solver_path)},
            "decisions": {
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return {
        **manifest,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compose(
            args.profile.resolve(),
            args.resolver_manifest.resolve(),
            args.output_dir.resolve(),
        )
    except (
        CompositionError,
        OfficialSourceError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

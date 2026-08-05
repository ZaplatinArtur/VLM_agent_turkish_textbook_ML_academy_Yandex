#!/usr/bin/env python3
"""Resolve a full-page fill-blank source under a frozen fail-closed profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.certificates import (  # noqa: E402
    answer_fingerprint,
    input_fingerprint,
    issue_certificate,
)
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
)
from evidence_os.fill_blank_page_activity import (  # noqa: E402
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


PROFILE_SCHEMA = "maxim-fill-blank-page-activity-profile-v1"
RUN_SCHEMA = "maxim-fill-blank-page-activity-run-v1"
CERTIFICATE_SCHEMA = "maxim-fill-blank-page-activity-certificate-v1"


class RunError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RunError(f"{path}: expected JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RunError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in output:
            raise RunError(f"{label}: missing/duplicate alignment key")
        output[task_id] = row
    return output


def _repo_path(configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RunError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _parse_documents(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        document_id, separator, raw_path = value.partition("=")
        document_id = document_id.strip()
        if not separator or not document_id or document_id in output:
            raise RunError("--document must be a unique DOCUMENT_ID=PATH pair")
        output[document_id] = Path(raw_path).resolve()
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def _validate_profile(profile: dict[str, Any]) -> None:
    if set(profile) != {
        "schema_version",
        "profile_name",
        "expected_rows",
        "anchor",
        "inputs",
        "documents",
        "runtime",
        "policy",
        "evaluation",
    } or profile.get("schema_version") != PROFILE_SCHEMA:
        raise RunError("fill-blank profile fields/schema are not allowlisted")
    if not isinstance(profile.get("profile_name"), str) or not profile["profile_name"]:
        raise RunError("fill-blank profile name is missing")
    if (
        not isinstance(profile.get("expected_rows"), int)
        or isinstance(profile["expected_rows"], bool)
        or profile["expected_rows"] < 1
    ):
        raise RunError("fill-blank expected_rows is malformed")
    anchor = profile.get("anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"path", "sha256", "role"}:
        raise RunError("fill-blank opaque anchor is malformed")
    if anchor.get("role") not in {
        "opaque_fail_closed_v6_anchor_only",
        "opaque_fail_closed_v7_main_anchor_only",
    }:
        raise RunError("fill-blank anchor role is not opaque")
    inputs = profile.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "parser_observations",
        "source_locators",
        "source_index",
    }:
        raise RunError("fill-blank profile inputs are malformed")
    roles = {
        "parser_observations": "observable_ocr_only",
        "source_locators": "public_source_identity_only",
        "source_index": "task_id_free_fill_blank_source_only",
    }
    for name, role in roles.items():
        spec = inputs.get(name)
        if (
            not isinstance(spec, dict)
            or set(spec) != {"path", "sha256", "allowed_role"}
            or spec.get("allowed_role") != role
        ):
            raise RunError(f"fill-blank {name} input role is malformed")
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "pypdf_version",
        "pdfplumber_version",
    }:
        raise RunError("fill-blank runtime pin is malformed")
    policy = profile.get("policy")
    required_policy = {
        "min_page_coverage",
        "min_page_matched_tokens",
        "min_page_margin",
        "min_activity_coverage",
        "min_activity_matched_tokens",
        "require_no_single_source_marker",
        "require_complete_item_inventory",
        "require_pdf_activity_title_once",
        "require_pdf_activity_instruction_once",
        "require_pdf_complete_item_inventory",
        "require_pdf_word_bank_multiset",
        "require_pdf_answer_key_components",
        "allow_generic_numberless_binding",
        "ambiguous_or_malformed_action",
        "task_id_is_policy_feature",
        "benchmark_candidate_or_outcome_access",
        "yandex_public_identity_projection",
    }
    if not isinstance(policy, dict) or set(policy) != required_policy:
        raise RunError("fill-blank policy fields are not allowlisted")
    expected_literals = {
        "require_no_single_source_marker": True,
        "require_complete_item_inventory": True,
        "require_pdf_activity_title_once": True,
        "require_pdf_activity_instruction_once": True,
        "require_pdf_complete_item_inventory": True,
        "require_pdf_word_bank_multiset": True,
        "require_pdf_answer_key_components": True,
        "allow_generic_numberless_binding": False,
        "ambiguous_or_malformed_action": "abstain_keep_anchor",
        "task_id_is_policy_feature": False,
        "benchmark_candidate_or_outcome_access": False,
        "yandex_public_identity_projection": (
            "url_name_plus_optional_numeric_nosw_v2"
        ),
    }
    if any(policy.get(key) != value for key, value in expected_literals.items()):
        raise RunError("fill-blank profile does not enable every fail-closed gate")
    documents = profile.get("documents")
    if not isinstance(documents, list) or not documents:
        raise RunError("fill-blank profile has no pinned documents")
    if not isinstance(profile.get("evaluation"), dict):
        raise RunError("fill-blank evaluation preregistration is missing")


def _certificate_record(task_id: str, result: Any, answer: str) -> dict[str, Any]:
    candidate = CandidateEnvelope(source=VERIFIER, final_answer=answer)
    certificate = issue_certificate(
        result.problem,
        candidate,
        kind=CertificateKind.SOURCE_ENTAILMENT,
        strength=CertificateStrength.STRONG,
        verdict=CertificateVerdict.PASS,
        verifier=VERIFIER,
        claim_coverage=1.0,
        contradiction_count=0,
        deterministic_checks=tuple(passed for _name, passed in result.checks),
        trace=canonical_json_bytes(result.trace),
    )
    return {
        "schema_version": CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "kind": certificate.kind.value,
        "strength": certificate.strength.value,
        "status": certificate.verdict.value,
        "input_fingerprint": input_fingerprint(result.problem),
        "answer_fingerprint": answer_fingerprint(candidate),
        "input_bound": True,
        "answer_bound": True,
        "claim_coverage": 1.0,
        "contradiction_count": 0,
        "deterministic_checks": list(certificate.deterministic_checks),
        "verifier": VERIFIER,
        "trace": result.trace,
        "trace_fingerprint": certificate.trace_fingerprint,
    }


def run(
    profile_path: Path,
    document_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    _validate_profile(profile)
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile["expected_rows"])
    inputs = profile["inputs"]
    policy = profile["policy"]
    runtime = profile["runtime"]
    parser_path = _repo_path(str(inputs["parser_observations"]["path"]))
    locator_path = _repo_path(str(inputs["source_locators"]["path"]))
    source_index_path = _repo_path(str(inputs["source_index"]["path"]))
    parser_sha = _require_hash(
        parser_path,
        str(inputs["parser_observations"]["sha256"]),
        "parser observations",
    )
    locator_sha = _require_hash(
        locator_path,
        str(inputs["source_locators"]["sha256"]),
        "source locators",
    )
    source_index_sha = _require_hash(
        source_index_path,
        str(inputs["source_index"]["sha256"]),
        "fill-blank source index",
    )
    index = parse_fill_blank_page_index(_load_json(source_index_path))
    indexed_documents = {document.document_id: document for document in index.documents}
    frozen_documents = profile["documents"]
    frozen_by_id = {
        str(item.get("document_id") or ""): item
        for item in frozen_documents
        if isinstance(item, dict)
    }
    if (
        len(frozen_by_id) != len(frozen_documents)
        or set(frozen_by_id) != set(indexed_documents)
        or set(document_paths) != set(indexed_documents)
    ):
        raise RunError("fill-blank indexed/frozen/provided document sets differ")
    try:
        import pdfplumber
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RunError("fill-blank resolver requires pinned PDF dependencies") from exc
    if (
        str(pdfplumber.__version__) != str(runtime["pdfplumber_version"])
        or str(pypdf.__version__) != str(runtime["pypdf_version"])
    ):
        raise RunError("fill-blank PDF runtime differs from frozen profile")
    thresholds = FillBlankPageThresholds(
        min_page_coverage=float(policy["min_page_coverage"]),
        min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
        min_page_margin=float(policy["min_page_margin"]),
        min_activity_coverage=float(policy["min_activity_coverage"]),
        min_activity_matched_tokens=int(policy["min_activity_matched_tokens"]),
    )
    caches: dict[str, dict[str, Any]] = {}
    for document_id, document in indexed_documents.items():
        frozen = frozen_by_id[document_id]
        if (
            set(frozen) != {"document_id", "pdf_sha256", "page_count"}
            or frozen.get("pdf_sha256") != document.pdf_sha256
            or frozen.get("page_count") != document.page_count
        ):
            raise RunError(f"fill-blank profile/index differ for {document_id}")
        pdf_path = document_paths[document_id]
        pdf_sha = _require_hash(pdf_path, document.pdf_sha256, "fill-blank PDF")
        reader = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if len(page_texts) != document.page_count:
            raise RunError("fill-blank PDF page count changed")
        caches[document_id] = {
            "path": pdf_path,
            "sha256": pdf_sha,
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
            "source_verification": verify_fill_blank_page_index_pdf(
                pdf_path, document
            ),
        }
    parser_rows = _load_jsonl(parser_path)
    if len(parser_rows) != expected_rows:
        raise RunError("fill-blank parser row count differs from profile")
    parser_index = _index(parser_rows, "parser")
    locator_index = _index(_load_jsonl(locator_path), "source locators")
    if not set(parser_index) <= set(locator_index):
        raise RunError("source locator projection does not cover parser rows")
    candidate_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    eligible = 0
    accepted = 0
    for raw in parser_rows:
        task_id = str(raw["task_id"])
        source_url = str(locator_index[task_id].get("source_url") or "")
        try:
            document = document_for_source(
                index, source_url, allow_missing_nosw=True
            )
        except OfficialSourceError:
            document = None
        if document is None:
            candidate_rows.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": "source_not_in_fill_blank_index",
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
            audit_rows.append(
                {
                    "task_id": task_id,
                    "eligible": False,
                    "accepted": False,
                    "reason": "source_not_in_fill_blank_index",
                }
            )
            continue
        eligible += 1
        cache = caches[document.document_id]
        try:
            observation = parser_observation_primary_layout_number(raw)
            result = resolve_fill_blank_page_activity(
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
        except (OfficialSourceError, ValueError, KeyError, TypeError) as exc:
            candidate_rows.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": str(exc),
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
            audit_rows.append(
                {
                    "task_id": task_id,
                    "eligible": True,
                    "accepted": False,
                    "reason": str(exc),
                }
            )
            continue
        result.trace["provenance"] = {
            "profile_sha256": profile_sha,
            "parser_observations_sha256": parser_sha,
            "source_locators_sha256": locator_sha,
            "source_index_sha256": source_index_sha,
            "fill_blank_pdf_sha256": cache["sha256"],
            "pypdf_version": str(pypdf.__version__),
            "pdfplumber_version": str(pdfplumber.__version__),
            "source_verification": cache["source_verification"],
        }
        if result.accepted and result.answer:
            accepted += 1
            candidate_rows.append(
                {
                    "task_id": task_id,
                    "final_answer": result.answer,
                    "abstain": False,
                    "error": None,
                    "generation": {
                        "gold_access": False,
                        "resolver": VERIFIER,
                        "source_certificate": True,
                    },
                }
            )
            certificate_rows.append(
                _certificate_record(task_id, result, result.answer)
            )
        else:
            failed = [name for name, passed in result.checks if not passed]
            candidate_rows.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": "failed_checks:" + ",".join(failed),
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
        audit_rows.append(
            {
                "task_id": task_id,
                "eligible": True,
                "accepted": result.accepted,
                "checks": {name: passed for name, passed in result.checks},
                "trace": result.trace,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate.jsonl"
    certificate_path = output_dir / "certificates.jsonl"
    audit_path = output_dir / "audit.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(certificate_path, certificate_rows)
    _write_jsonl(audit_path, audit_rows)
    manifest = {
        "schema_version": RUN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "task_id_used_for_alignment_only": True,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "rows": expected_rows,
        "eligible_rows": eligible,
        "accepted_certificates": accepted,
        "abstentions": expected_rows - accepted,
        "artifacts": {
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "certificates": {
                "path": str(certificate_path),
                "sha256": sha256_file(certificate_path),
            },
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        },
        "inputs": {
            "parser_observations": {"path": str(parser_path), "sha256": parser_sha},
            "source_locators": {"path": str(locator_path), "sha256": locator_sha},
            "source_index": {
                "path": str(source_index_path),
                "sha256": source_index_sha,
            },
            "documents": {
                document_id: {
                    "path": str(cache["path"]),
                    "sha256": cache["sha256"],
                }
                for document_id, cache in sorted(caches.items())
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
    parser.add_argument("--document", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.profile.resolve(),
            _parse_documents(args.document),
            args.output_dir.resolve(),
        )
    except (
        RunError,
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

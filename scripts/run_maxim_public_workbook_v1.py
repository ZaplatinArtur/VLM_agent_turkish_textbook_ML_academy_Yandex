#!/usr/bin/env python3
"""Resolve public workbook questions under a frozen task-ID-free source index."""

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

from evidence_os.certificates import answer_fingerprint, input_fingerprint, issue_certificate  # noqa: E402
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
)
from evidence_os.official_ogm import (  # noqa: E402
    PageMatcher,
    canonical_json_bytes,
    parser_observation_allow_missing_number,
    parser_observation_primary_layout_number,
    sha256_file,
)
from evidence_os.official_workbook import (  # noqa: E402
    VERIFIER,
    WorkbookThresholds,
    document_for_source,
    parse_workbook_index,
    resolve_workbook_question,
    verify_workbook_index_pdf,
)
from evidence_os.official_ogm import OfficialSourceError  # noqa: E402


SCHEMA = "maxim-public-workbook-run-v1"


class RunError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RunError(f"{path}: expected object")
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
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in result:
            raise RunError(f"{label}: missing/duplicate task_id")
        result[task_id] = row
    return result


def _repo_path(configured: str) -> Path:
    raw = Path(configured)
    return raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RunError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _parse_documents(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        document_id, separator, raw_path = value.partition("=")
        document_id = document_id.strip()
        if not separator or not document_id or document_id in result:
            raise RunError("--document must be a unique DOCUMENT_ID=PATH pair")
        result[document_id] = Path(raw_path).resolve()
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


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
        deterministic_checks=tuple(passed for _, passed in result.checks),
        trace=canonical_json_bytes(result.trace),
    )
    return {
        "schema_version": "maxim-public-workbook-certificate-v1",
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
    if profile.get("schema_version") != "maxim-public-workbook-profile-v1":
        raise RunError("unsupported public-workbook profile schema")
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile.get("expected_rows", 0))
    inputs = profile.get("inputs")
    policy = profile.get("policy")
    runtime = profile.get("runtime")
    frozen_documents = profile.get("documents")
    if (
        expected_rows < 1
        or not isinstance(inputs, dict)
        or not isinstance(policy, dict)
        or not isinstance(frozen_documents, list)
        or not isinstance(runtime, dict)
    ):
        raise RunError("public-workbook profile is incomplete")
    parser_spec = inputs.get("parser_observations")
    locator_spec = inputs.get("source_locators")
    index_spec = inputs.get("source_index")
    if not all(isinstance(item, dict) for item in (parser_spec, locator_spec, index_spec)):
        raise RunError("public-workbook inputs must pin parser, locators, and source index")
    parser_path = _repo_path(str(parser_spec["path"]))
    locator_path = _repo_path(str(locator_spec["path"]))
    source_index_path = _repo_path(str(index_spec["path"]))
    parser_sha = _require_hash(parser_path, str(parser_spec["sha256"]), "parser observations")
    locator_sha = _require_hash(locator_path, str(locator_spec["sha256"]), "source locators")
    source_index_sha = _require_hash(
        source_index_path, str(index_spec["sha256"]), "source-native workbook index"
    )
    source_index = parse_workbook_index(_load_json(source_index_path))
    indexed_documents = {document.document_id: document for document in source_index.documents}
    frozen_by_id: dict[str, dict[str, Any]] = {}
    for raw in frozen_documents:
        if not isinstance(raw, dict):
            raise RunError("frozen workbook document entry is malformed")
        document_id = str(raw.get("document_id") or "")
        if not document_id or document_id in frozen_by_id:
            raise RunError("frozen workbook document IDs must be unique")
        frozen_by_id[document_id] = raw
    if set(frozen_by_id) != set(indexed_documents) or set(document_paths) != set(indexed_documents):
        raise RunError("provided, frozen, and indexed workbook document sets must match exactly")
    thresholds = WorkbookThresholds(
        min_page_coverage=float(policy["min_page_coverage"]),
        min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
        min_page_margin=float(policy["min_page_margin"]),
        min_numberless_question_coverage=float(policy["min_numberless_question_coverage"]),
        min_numberless_question_matched_tokens=int(
            policy["min_numberless_question_matched_tokens"]
        ),
        min_numberless_question_margin=float(policy["min_numberless_question_margin"]),
    )
    try:
        import pdfplumber
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RunError("public-workbook resolver requires the official-source dependency extra") from exc
    if (
        str(runtime.get("pypdf_version") or "") != str(pypdf.__version__)
        or str(runtime.get("pdfplumber_version") or "") != str(pdfplumber.__version__)
    ):
        raise RunError("public-workbook PDF runtime differs from the frozen profile")
    if (
        policy.get("require_observed_question_number") is not True
        or policy.get("allow_numberless_question_binding") is not False
        or policy.get("require_unique_printed_number_on_page") is not True
        or policy.get("require_pdf_bound_key_context") is not True
    ):
        raise RunError("public-workbook profile does not enable all fail-closed bindings")
    number_projection = str(
        policy.get("question_number_projection") or "unique_block_markers_v1"
    )
    if number_projection == "unique_block_markers_v1":
        observation_loader = parser_observation_allow_missing_number
    elif number_projection == "primary_layout_then_unique_v1":
        observation_loader = parser_observation_primary_layout_number
    else:
        raise RunError("unsupported question-number projection")
    identity_projection = str(
        policy.get("yandex_public_identity_projection")
        or "url_name_plus_required_numeric_nosw_v1"
    )
    if identity_projection not in {
        "url_name_plus_required_numeric_nosw_v1",
        "url_name_plus_optional_numeric_nosw_v2",
    }:
        raise RunError("unsupported Yandex public-identity projection")
    allow_missing_nosw = identity_projection == "url_name_plus_optional_numeric_nosw_v2"

    caches: dict[str, dict[str, Any]] = {}
    for document_id, document in indexed_documents.items():
        frozen = frozen_by_id[document_id]
        if (
            str(frozen.get("pdf_sha256") or "") != document.pdf_sha256
            or int(frozen.get("page_count", 0)) != document.page_count
        ):
            raise RunError(f"profile and source index disagree for {document_id}")
        pdf_path = document_paths[document_id]
        pdf_sha = _require_hash(pdf_path, document.pdf_sha256, f"workbook PDF {document_id}")
        reader = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if len(page_texts) != document.page_count:
            raise RunError(f"workbook PDF {document_id} page count changed")
        caches[document_id] = {
            "path": pdf_path,
            "sha256": pdf_sha,
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
            "source_verification": verify_workbook_index_pdf(pdf_path, document),
        }

    parser_rows = _load_jsonl(parser_path)
    if len(parser_rows) != expected_rows:
        raise RunError(f"parser must contain exactly {expected_rows} rows")
    parser_index = _index(parser_rows, "parser")
    locator_index = _index(_load_jsonl(locator_path), "source locators")
    if not set(parser_index) <= set(locator_index):
        raise RunError("source locator projection does not cover every parser task")

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
                source_index,
                source_url,
                allow_missing_nosw=allow_missing_nosw,
            )
        except OfficialSourceError:
            document = None
        if document is None:
            candidate_rows.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": "source_not_in_public_workbook_index",
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
            audit_rows.append(
                {
                    "task_id": task_id,
                    "eligible": False,
                    "accepted": False,
                    "reason": "source_not_in_public_workbook_index",
                }
            )
            continue
        eligible += 1
        cache = caches[document.document_id]
        try:
            observation = observation_loader(raw)
            result = resolve_workbook_question(
                observation,
                source_url,
                document,
                cache["matcher"],
                cache["page_texts"],
                thresholds,
                allow_missing_nosw=allow_missing_nosw,
                verified_content_marker_counts=cache["source_verification"][
                    "content_marker_counts"
                ],
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
            "workbook_pdf_sha256": cache["sha256"],
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
            certificate_rows.append(_certificate_record(task_id, result, result.answer))
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
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "task_id_used_for_alignment_only": True,
        "viewer_nosw_used_as_policy_feature": False,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "rows": expected_rows,
        "eligible_rows": eligible,
        "accepted_certificates": accepted,
        "abstentions": expected_rows - accepted,
        "artifacts": {
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "certificates": {
                "path": str(certificate_path),
                "sha256": sha256_file(certificate_path),
            },
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        },
        "inputs": {
            "parser_observations": {"path": str(parser_path), "sha256": parser_sha},
            "source_locators": {"path": str(locator_path), "sha256": locator_sha},
            "source_index": {"path": str(source_index_path), "sha256": source_index_sha},
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
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    manifest["manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--document", action="append", default=[], metavar="DOCUMENT_ID=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.profile_json.resolve(),
            _parse_documents(args.document),
            args.output_dir.resolve(),
        )
    except (RunError, OfficialSourceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

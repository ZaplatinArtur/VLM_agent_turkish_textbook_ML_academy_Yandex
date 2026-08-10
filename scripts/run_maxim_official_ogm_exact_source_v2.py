#!/usr/bin/env python3
"""Resolve exact OGM questions under a frozen, score-blind profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from evidence_os.official_ogm import (  # noqa: E402
    VERIFIER,
    MatchThresholds,
    OfficialSourceError,
    PageMatcher,
    build_safe_snapshot,
    canonical_json_bytes,
    canonical_json_sha256,
    parser_observation,
    resolve_exact_question,
    sha256_file,
    strict_book_id,
)


SCHEMA = "maxim-official-ogm-exact-source-run-v2"


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
                raise RunError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _index_unique(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in indexed:
            raise RunError(f"{source}: missing/duplicate task_id")
        indexed[task_id] = row
    return indexed


def _resolve_profile_path(profile_path: Path, configured: str) -> Path:
    raw = Path(configured)
    if raw.is_absolute():
        return raw.resolve()
    # Repository profiles use repository-relative paths, not config-relative paths.
    return (REPO_ROOT / raw).resolve()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RunError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _extract_pdf_text(pdf_path: Path) -> tuple[list[str], str]:
    try:
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RunError("official OGM resolver requires pypdf; install the official-source extra") from exc
    reader = PdfReader(str(pdf_path))
    texts = [page.extract_text() or "" for page in reader.pages]
    if not texts or all(not text.strip() for text in texts):
        raise RunError("official PDF has no extractable text")
    return texts, str(pypdf.__version__)


def _certificate_record(task_id: str, result: Any, answer: str) -> dict[str, Any]:
    trace_bytes = canonical_json_bytes(result.trace)
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
        deterministic_checks=tuple(value for _, value in result.checks),
        trace=trace_bytes,
    )
    return {
        "schema_version": "maxim-official-ogm-certificate-v2",
        "task_id": task_id,
        "kind": certificate.kind.value,
        "strength": certificate.strength.value,
        "status": certificate.verdict.value,
        "input_fingerprint": input_fingerprint(result.problem),
        "answer_fingerprint": answer_fingerprint(candidate),
        "input_bound": certificate.input_bound,
        "answer_bound": certificate.answer_bound,
        "claim_coverage": certificate.claim_coverage,
        "contradiction_count": certificate.contradiction_count,
        "deterministic_checks": list(certificate.deterministic_checks),
        "verifier": certificate.verifier,
        "trace": result.trace,
        "trace_fingerprint": certificate.trace_fingerprint,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def run(
    profile_path: Path,
    book_json: Path,
    tests_dir: Path,
    pdf_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    if profile.get("schema_version") != "maxim-official-ogm-exact-source-profile-v2":
        raise RunError("unsupported or missing resolver profile schema")
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile.get("expected_rows", 0))
    if expected_rows < 1:
        raise RunError("profile expected_rows must be positive")
    inputs = profile.get("inputs")
    official = profile.get("official")
    policy = profile.get("policy")
    if not isinstance(inputs, dict) or not isinstance(official, dict) or not isinstance(policy, dict):
        raise RunError("profile inputs/official/policy sections are required")
    parser_spec = inputs.get("parser_observations")
    locator_spec = inputs.get("source_locators")
    if not isinstance(parser_spec, dict) or not isinstance(locator_spec, dict):
        raise RunError("profile must pin parser observations and source locators")
    parser_path = _resolve_profile_path(profile_path, str(parser_spec.get("path") or ""))
    locator_path = _resolve_profile_path(profile_path, str(locator_spec.get("path") or ""))
    parser_sha = _require_hash(
        parser_path, str(parser_spec.get("sha256") or ""), "parser observations"
    )
    locator_sha = _require_hash(
        locator_path, str(locator_spec.get("sha256") or ""), "source locators"
    )
    expected_book_id = str(official.get("book_id") or "")
    if strict_book_id(f"https://ogmmateryal.eba.gov.tr/ogm-test/book/{expected_book_id}") != expected_book_id:
        raise RunError("profile OGM book ID is malformed")
    book_payload = _load_json(book_json)
    test_payloads = [_load_json(path) for path in sorted(tests_dir.glob("test_*.json"))]
    snapshot = build_safe_snapshot(book_payload, test_payloads)
    snapshot_sha = canonical_json_sha256(snapshot)
    if snapshot["book"]["id"] != expected_book_id:
        raise RunError("official snapshot is for a different book")
    if snapshot_sha != str(official.get("safe_snapshot_sha256") or ""):
        raise RunError(
            "official safe snapshot changed; freeze a new profile before evaluating it"
        )
    pdf_sha = _require_hash(pdf_path, str(official.get("pdf_sha256") or ""), "official PDF")
    if str(snapshot["book"]["pdfPublicUrl"]) != str(official.get("pdf_url") or ""):
        raise RunError("profile PDF URL differs from the safe official snapshot")
    thresholds = MatchThresholds(
        min_idf_coverage=float(policy.get("min_idf_coverage")),
        min_matched_tokens=int(policy.get("min_matched_tokens")),
        min_page_margin=float(policy.get("min_page_margin")),
        min_candidate_margin=float(policy.get("min_candidate_margin")),
        max_aspect_log_delta=float(policy.get("max_aspect_log_delta")),
        pdf_page_index_offset=int(policy.get("pdf_page_index_offset")),
    )
    pdf_texts, pypdf_version = _extract_pdf_text(pdf_path)
    matcher = PageMatcher(pdf_texts)
    parser_rows = _load_jsonl(parser_path)
    if len(parser_rows) != expected_rows:
        raise RunError(f"parser must contain exactly {expected_rows} rows")
    parser_index = _index_unique(parser_rows, "parser observations")
    locator_index = _index_unique(_load_jsonl(locator_path), "source locators")
    if not set(parser_index) <= set(locator_index):
        raise RunError("source locator projection does not cover every parser task")

    candidates: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    eligible = 0
    accepted = 0
    for raw in parser_rows:
        task_id = str(raw["task_id"])
        source_url = str(locator_index[task_id].get("source_url") or "")
        try:
            source_book_id = strict_book_id(source_url)
        except OfficialSourceError:
            source_book_id = None
        if source_book_id != expected_book_id:
            candidates.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": "source_not_target_official_ogm_book",
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
            audit.append(
                {"task_id": task_id, "eligible": False, "accepted": False, "reason": "source_not_target_official_ogm_book"}
            )
            continue
        eligible += 1
        try:
            observation = parser_observation(raw)
            result = resolve_exact_question(
                observation, source_url, snapshot, matcher, thresholds
            )
        except (OfficialSourceError, ValueError, KeyError, TypeError) as exc:
            candidates.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": str(exc),
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
            audit.append(
                {"task_id": task_id, "eligible": True, "accepted": False, "reason": str(exc)}
            )
            continue
        # Bind the trace to every frozen source artifact before issuing it.
        result.trace["provenance"] = {
            "profile_sha256": profile_sha,
            "parser_observations_sha256": parser_sha,
            "source_locators_sha256": locator_sha,
            "official_safe_snapshot_sha256": snapshot_sha,
            "official_pdf_sha256": pdf_sha,
            "pypdf_version": pypdf_version,
        }
        if result.accepted and result.answer:
            accepted += 1
            answer = result.answer
            candidates.append(
                {
                    "task_id": task_id,
                    "final_answer": answer,
                    "abstain": False,
                    "error": None,
                    "generation": {
                        "gold_access": False,
                        "resolver": VERIFIER,
                        "source_certificate": True,
                    },
                }
            )
            certificates.append(_certificate_record(task_id, result, answer))
        else:
            failed = [name for name, passed in result.checks if not passed]
            candidates.append(
                {
                    "task_id": task_id,
                    "final_answer": "",
                    "abstain": True,
                    "error": "failed_checks:" + ",".join(failed),
                    "generation": {"gold_access": False, "resolver": VERIFIER},
                }
            )
        audit.append(
            {
                "task_id": task_id,
                "eligible": True,
                "accepted": bool(result.accepted),
                "checks": {name: passed for name, passed in result.checks},
                "trace": result.trace,
            }
        )

    if len(candidates) != expected_rows:
        raise RunError("candidate output lost task alignment")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate.jsonl"
    certificate_path = output_dir / "certificates.jsonl"
    audit_path = output_dir / "audit.jsonl"
    snapshot_path = output_dir / "official_safe_snapshot.json"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(certificate_path, certificates)
    _write_jsonl(audit_path, audit)
    snapshot_path.write_bytes(canonical_json_bytes(snapshot) + b"\n")
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "task_id_used_for_alignment_only": True,
        "source_url_used_only_as_evidence_locator": True,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "rows": expected_rows,
        "eligible_rows": eligible,
        "accepted_certificates": accepted,
        "abstentions": expected_rows - accepted,
        "artifacts": {
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "certificates": {"path": str(certificate_path), "sha256": sha256_file(certificate_path)},
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "official_safe_snapshot": {"path": str(snapshot_path), "sha256": sha256_file(snapshot_path)},
        },
        "inputs": {
            "parser_observations": {"path": str(parser_path), "sha256": parser_sha},
            "source_locators": {"path": str(locator_path), "sha256": locator_sha},
            "official_safe_snapshot_sha256": snapshot_sha,
            "official_pdf": {"path": str(pdf_path), "sha256": pdf_sha},
            "book_raw_sha256": sha256_file(book_json),
            "test_raw_sha256": canonical_json_sha256(
                sorted(sha256_file(path) for path in tests_dir.glob("test_*.json"))
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--book-json", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.profile_json.resolve(),
            args.book_json.resolve(),
            args.tests_dir.resolve(),
            args.pdf.resolve(),
            args.output_dir.resolve(),
        )
    except (RunError, OfficialSourceError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

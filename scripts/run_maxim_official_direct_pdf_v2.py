#!/usr/bin/env python3
"""Resolve pinned official-PDF questions under a frozen score-blind profile."""

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
    parser_observation,
    sha256_file,
)
from evidence_os.official_pdf import (  # noqa: E402
    VERIFIER,
    DirectPdfThresholds,
    OfficialSourceError,
    parse_key_regions,
    resolve_direct_pdf_question,
)


SCHEMA = "maxim-official-direct-pdf-run-v2"


class RunError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RunError(f"{path}: expected object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    result = {}
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
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or name.strip() in result:
            raise RunError("--document must be a unique NAME=PATH pair")
        result[name.strip()] = Path(raw_path).resolve()
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
        deterministic_checks=tuple(value for _, value in result.checks),
        trace=canonical_json_bytes(result.trace),
    )
    return {
        "schema_version": "maxim-official-direct-pdf-certificate-v2",
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
    if profile.get("schema_version") != "maxim-official-direct-pdf-profile-v2":
        raise RunError("unsupported direct-PDF profile schema")
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile.get("expected_rows", 0))
    inputs = profile.get("inputs")
    documents = profile.get("documents")
    policy = profile.get("policy")
    if expected_rows < 1 or not isinstance(inputs, dict) or not isinstance(documents, list) or not isinstance(policy, dict):
        raise RunError("direct-PDF profile is incomplete")
    parser_spec = inputs["parser_observations"]
    locator_spec = inputs["source_locators"]
    parser_path = _repo_path(str(parser_spec["path"]))
    locator_path = _repo_path(str(locator_spec["path"]))
    parser_sha = _require_hash(parser_path, str(parser_spec["sha256"]), "parser observations")
    locator_sha = _require_hash(locator_path, str(locator_spec["sha256"]), "source locators")
    thresholds = DirectPdfThresholds(
        min_page_coverage=float(policy["min_page_coverage"]),
        min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
        min_page_margin=float(policy["min_page_margin"]),
        rescue_min_page_coverage=float(policy["rescue_min_page_coverage"]),
        rescue_min_page_margin=float(policy["rescue_min_page_margin"]),
        rescue_min_anchor_coverage=float(policy["rescue_min_anchor_coverage"]),
        rescue_min_anchor_matched_tokens=int(policy["rescue_min_anchor_matched_tokens"]),
        rescue_min_anchor_margin=float(policy["rescue_min_anchor_margin"]),
    )
    try:
        import pdfplumber
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RunError("direct-PDF resolver requires the official-source dependency extra") from exc

    configured_names = {str(item.get("name") or "") for item in documents if isinstance(item, dict)}
    if (
        not configured_names
        or len(configured_names) != len(documents)
        or configured_names != set(document_paths)
    ):
        raise RunError("--document names must exactly match the frozen profile")
    adapters_by_url: dict[str, dict[str, Any]] = {}
    caches: dict[str, dict[str, Any]] = {}
    for raw_adapter in documents:
        if not isinstance(raw_adapter, dict):
            raise RunError("direct-PDF adapter is malformed")
        adapter = dict(raw_adapter)
        name = str(adapter["name"])
        source_url = str(adapter["source_url"])
        if source_url in adapters_by_url:
            raise RunError("direct-PDF source URL is duplicated")
        pdf_path = document_paths[name]
        pdf_sha = _require_hash(pdf_path, str(adapter["pdf_sha256"]), f"official PDF {name}")
        reader = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if len(page_texts) != int(adapter["page_count"]):
            raise RunError(f"official PDF {name} page count changed")
        answer_key = parse_key_regions(
            pdf_path,
            key_page_number=int(adapter["key_page_number"]),
            regions=adapter["key_regions"],
        )
        adapters_by_url[source_url] = adapter
        caches[source_url] = {
            "path": pdf_path,
            "sha256": pdf_sha,
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
            "answer_key": answer_key,
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
        adapter = adapters_by_url.get(source_url)
        if adapter is None:
            candidate_rows.append(
                {"task_id": task_id, "final_answer": "", "abstain": True, "error": "source_not_pinned_direct_pdf", "generation": {"gold_access": False, "resolver": VERIFIER}}
            )
            audit_rows.append({"task_id": task_id, "eligible": False, "accepted": False, "reason": "source_not_pinned_direct_pdf"})
            continue
        eligible += 1
        cache = caches[source_url]
        try:
            observation = parser_observation(raw)
            result = resolve_direct_pdf_question(
                observation,
                source_url,
                adapter,
                cache["matcher"],
                cache["page_texts"],
                cache["answer_key"],
                thresholds,
            )
        except (OfficialSourceError, ValueError, KeyError, TypeError) as exc:
            candidate_rows.append(
                {"task_id": task_id, "final_answer": "", "abstain": True, "error": str(exc), "generation": {"gold_access": False, "resolver": VERIFIER}}
            )
            audit_rows.append({"task_id": task_id, "eligible": True, "accepted": False, "reason": str(exc)})
            continue
        result.trace["provenance"] = {
            "profile_sha256": profile_sha,
            "parser_observations_sha256": parser_sha,
            "source_locators_sha256": locator_sha,
            "official_pdf_sha256": cache["sha256"],
            "pypdf_version": str(pypdf.__version__),
            "pdfplumber_version": str(pdfplumber.__version__),
        }
        if result.accepted and result.answer:
            accepted += 1
            candidate_rows.append(
                {"task_id": task_id, "final_answer": result.answer, "abstain": False, "error": None, "generation": {"gold_access": False, "resolver": VERIFIER, "source_certificate": True}}
            )
            certificate_rows.append(_certificate_record(task_id, result, result.answer))
        else:
            failed = [name for name, passed in result.checks if not passed]
            candidate_rows.append(
                {"task_id": task_id, "final_answer": "", "abstain": True, "error": "failed_checks:" + ",".join(failed), "generation": {"gold_access": False, "resolver": VERIFIER}}
            )
        audit_rows.append(
            {"task_id": task_id, "eligible": True, "accepted": result.accepted, "checks": {name: passed for name, passed in result.checks}, "trace": result.trace}
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
        },
        "inputs": {
            "parser_observations": {"path": str(parser_path), "sha256": parser_sha},
            "source_locators": {"path": str(locator_path), "sha256": locator_sha},
            "documents": {
                adapter["name"]: {"path": str(caches[url]["path"]), "sha256": caches[url]["sha256"]}
                for url, adapter in adapters_by_url.items()
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--document", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.profile_json.resolve(), _parse_documents(args.document), args.output_dir.resolve())
    except (RunError, OfficialSourceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

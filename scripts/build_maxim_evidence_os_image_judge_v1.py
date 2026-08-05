#!/usr/bin/env python3
"""Re-adjudicate changed image rows from pre-score official-source certificates."""

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

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.adapters import certificate_from_record  # noqa: E402
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
)
from evidence_os.official_ogm import (  # noqa: E402
    OfficialSourceError,
    PageMatcher,
    parser_observation_allow_missing_number,
    problem_for,
)
from evidence_os.official_workbook import (  # noqa: E402
    VERIFIER as WORKBOOK_VERIFIER,
    WorkbookThresholds,
    document_for_source,
    parse_workbook_index,
    resolve_workbook_question,
    verify_workbook_index_pdf,
)
from compose_maxim_official_ogm_failclosed_v2 import (  # noqa: E402
    CompositionError,
    _validate_workbook_certificate_artifact,
)


SCHEMA = "maxim-evidence-os-image-judge-v1"


class JudgeBuildError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise JudgeBuildError(f"{path}: expected an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise JudgeBuildError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise JudgeBuildError(f"{label}: missing or duplicate task_id")
        result[task_id] = row
    return result


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise JudgeBuildError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build(
    profile_path: Path,
    resolver_manifest_path: Path,
    expected_resolver_manifest_sha256: str,
    composition_manifest_path: Path,
    expected_composition_manifest_sha256: str,
    base_solver_path: Path,
    expected_base_solver_sha256: str,
    base_judge_path: Path,
    expected_base_judge_sha256: str,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    resolver_manifest = _load_json(resolver_manifest_path)
    composition_manifest = _load_json(composition_manifest_path)
    profile_sha = sha256_file(profile_path)
    resolver_sha = _require_hash(
        resolver_manifest_path,
        expected_resolver_manifest_sha256,
        "resolver manifest",
    )
    composition_sha = _require_hash(
        composition_manifest_path,
        expected_composition_manifest_sha256,
        "composition manifest",
    )
    if profile.get("schema_version") != "maxim-public-workbook-profile-v1":
        raise JudgeBuildError("image adjudication requires the public-workbook profile")
    if (
        resolver_manifest.get("schema_version") != "maxim-public-workbook-run-v1"
        or composition_manifest.get("schema_version")
        != "maxim-official-source-failclosed-composition-v2"
        or resolver_manifest.get("gold_access") is not False
        or composition_manifest.get("profile", {}).get("sha256") != profile_sha
        or composition_manifest.get("resolver_manifest", {}).get("sha256") != resolver_sha
        or composition_manifest.get("score_or_outcome_access") is not False
        or resolver_manifest.get("benchmark_candidate_or_outcome_access") is not False
    ):
        raise JudgeBuildError("composition/resolver is not bound to the frozen blind profile")
    base_solver_sha = _require_hash(
        base_solver_path, expected_base_solver_sha256, "base image-judge solver"
    )
    base_judge_sha = _require_hash(
        base_judge_path, expected_base_judge_sha256, "base image judge"
    )
    solver_spec = composition_manifest.get("output", {}).get("solver")
    decisions_spec = composition_manifest.get("output", {}).get("decisions")
    artifacts = resolver_manifest.get("artifacts")
    if not all(isinstance(item, dict) for item in (solver_spec, decisions_spec, artifacts)):
        raise JudgeBuildError("composition or resolver artifacts are incomplete")
    solver_path = Path(str(solver_spec["path"])).resolve()
    decisions_path = Path(str(decisions_spec["path"])).resolve()
    candidate_path = Path(str(artifacts["candidate"]["path"])).resolve()
    certificate_path = Path(str(artifacts["certificates"]["path"])).resolve()
    _require_hash(solver_path, str(solver_spec["sha256"]), "composed solver")
    _require_hash(decisions_path, str(decisions_spec["sha256"]), "composition decisions")
    _require_hash(candidate_path, str(artifacts["candidate"]["sha256"]), "resolver candidate")
    _require_hash(
        certificate_path, str(artifacts["certificates"]["sha256"]), "resolver certificates"
    )
    solver = _index(_load_jsonl(solver_path), "composed solver")
    base_solver = _index(_load_jsonl(base_solver_path), "base solver")
    anchor_spec = profile.get("anchor")
    if not isinstance(anchor_spec, dict):
        raise JudgeBuildError("profile anchor is missing")
    anchor_path = (REPO_ROOT / str(anchor_spec.get("path") or "")).resolve()
    _require_hash(anchor_path, str(anchor_spec.get("sha256") or ""), "profile anchor")
    anchor = _index(_load_jsonl(anchor_path), "profile anchor")
    decisions = _index(_load_jsonl(decisions_path), "composition decisions")
    candidates = _index(_load_jsonl(candidate_path), "resolver candidate")
    certificates = _index(_load_jsonl(certificate_path), "resolver certificates")
    parser_spec = profile["inputs"]["parser_observations"]
    locator_spec = profile["inputs"]["source_locators"]
    source_index_spec = profile["inputs"]["source_index"]
    parser_path = (REPO_ROOT / str(parser_spec["path"])).resolve()
    locator_path = (REPO_ROOT / str(locator_spec["path"])).resolve()
    source_index_path = (REPO_ROOT / str(source_index_spec["path"])).resolve()
    _require_hash(parser_path, str(parser_spec["sha256"]), "parser observations")
    _require_hash(locator_path, str(locator_spec["sha256"]), "source locators")
    _require_hash(source_index_path, str(source_index_spec["sha256"]), "source index")
    parser_rows = _index(_load_jsonl(parser_path), "parser observations")
    locator_rows = _index(_load_jsonl(locator_path), "source locators")
    source_index = parse_workbook_index(_load_json(source_index_path))
    source_documents = {document.document_id: document for document in source_index.documents}
    policy = profile.get("policy")
    runtime = profile.get("runtime")
    if not isinstance(policy, dict) or not isinstance(runtime, dict):
        raise JudgeBuildError("profile policy/runtime is incomplete")
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
        raise JudgeBuildError("certificate adjudication requires pinned PDF runtimes") from exc
    if (
        str(runtime.get("pypdf_version") or "") != str(pypdf.__version__)
        or str(runtime.get("pdfplumber_version") or "") != str(pdfplumber.__version__)
    ):
        raise JudgeBuildError("PDF runtime differs from the frozen profile")
    manifest_documents = resolver_manifest.get("inputs", {}).get("documents")
    if not isinstance(manifest_documents, dict) or set(manifest_documents) != set(source_documents):
        raise JudgeBuildError("resolver document set differs from the pinned source index")
    source_verification: dict[str, dict[str, int]] = {}
    source_caches: dict[str, dict[str, Any]] = {}
    for document_id, document in source_documents.items():
        document_spec = manifest_documents[document_id]
        if not isinstance(document_spec, dict):
            raise JudgeBuildError(f"resolver document {document_id} is malformed")
        pdf_path = Path(str(document_spec.get("path") or "")).resolve()
        _require_hash(pdf_path, document.pdf_sha256, f"workbook PDF {document_id}")
        source_verification[document_id] = verify_workbook_index_pdf(pdf_path, document)
        page_texts = [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]
        if len(page_texts) != document.page_count:
            raise JudgeBuildError(f"workbook PDF {document_id} page count changed")
        source_caches[document_id] = {
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
        }
    base_judge_rows = _load_jsonl(base_judge_path)
    base_judge = _index(base_judge_rows, "base image judge")
    if (
        len(solver) != 274
        or len(anchor) != 274
        or len(base_solver) != 274
        or len(base_judge) != 97
    ):
        raise JudgeBuildError("expected composed/base solver and image judge rows 274/274/97")
    if (
        set(solver) != set(anchor)
        or set(decisions) != set(solver)
        or set(candidates) != set(solver)
        or set(parser_rows) != set(solver)
        or not set(solver) <= set(locator_rows)
        or not set(certificates) <= set(solver)
    ):
        raise JudgeBuildError("composed/base solver and decision task sets do not align")
    if not set(base_judge) <= set(solver):
        raise JudgeBuildError("image judge contains an unknown task")
    for task_id, decision in decisions.items():
        base_answer = str(base_solver[task_id].get("final_answer") or "").strip()
        anchor_answer = str(anchor[task_id].get("final_answer") or "").strip()
        current_answer = str(solver[task_id].get("final_answer") or "").strip()
        action = decision.get("action")
        if (
            decision.get("anchor_answer") != anchor_answer
            or decision.get("selected_answer") != current_answer
            or action not in {"keep_anchor", "replace_anchor"}
            or (action == "keep_anchor" and current_answer != anchor_answer)
            or (action == "replace_anchor" and current_answer == anchor_answer)
        ):
            raise JudgeBuildError(f"decision {task_id} is inconsistent with solver artifacts")
        if task_id in base_judge and anchor_answer != base_answer:
            raise JudgeBuildError(
                f"image row {task_id} anchor differs from the candidate judged by base image judge"
            )
    changed_image_ids = {
        task_id
        for task_id, decision in decisions.items()
        if decision.get("action") == "replace_anchor" and task_id in base_judge
    }
    if not changed_image_ids:
        raise JudgeBuildError("no changed image rows require certificate adjudication")
    output_rows: list[dict[str, Any]] = []
    adjudicated: list[dict[str, Any]] = []
    for original in base_judge_rows:
        task_id = str(original["task_id"])
        current_answer = str(solver[task_id].get("final_answer") or "").strip()
        if task_id not in changed_image_ids:
            base_answer = str(base_solver[task_id].get("final_answer") or "").strip()
            if current_answer != base_answer:
                raise JudgeBuildError(
                    f"unchanged image row {task_id} differs from the base judge candidate"
                )
            output_rows.append(dict(original))
            continue
        candidate = candidates.get(task_id)
        certificate = certificates.get(task_id)
        if candidate is None or certificate is None:
            raise JudgeBuildError(f"changed image row {task_id} lacks its source certificate")
        trace = certificate.get("trace")
        if not isinstance(trace, dict):
            raise JudgeBuildError(f"changed image row {task_id} lacks an inline trace")
        try:
            _validate_workbook_certificate_artifact(
                task_id=task_id,
                raw_candidate=candidate,
                raw_certificate=certificate,
                trace=trace,
                profile=profile,
                profile_sha=profile_sha,
            )
        except CompositionError as exc:
            raise JudgeBuildError(str(exc)) from exc
        candidate_answer = str(candidate.get("final_answer") or "").strip()
        trace_source = trace.get("source")
        if not isinstance(trace_source, dict):
            raise JudgeBuildError(f"changed image row {task_id} lacks source metadata")
        observation = parser_observation_allow_missing_number(parser_rows[task_id])
        source_url = str(locator_rows[task_id].get("source_url") or "")
        document = document_for_source(source_index, source_url)
        if document is None:
            raise JudgeBuildError(f"changed image row {task_id} has no indexed source document")
        source_record_id = str(trace_source.get("record_id") or "")
        source_records = {
            question.record_id: question
            for question in document.questions
        }
        source_record = source_records.get(source_record_id)
        if (
            source_record is None
            or trace_source.get("document_id") != document.document_id
            or trace_source.get("public_locator") != document.identity.public_locator
            or trace_source.get("name") != document.identity.name
            or trace_source.get("pdf_sha256") != document.pdf_sha256
            or trace_source.get("question_number") != source_record.question_number
            or trace_source.get("matched_page_number") != source_record.content_page_number
            or trace_source.get("key_page_number") != source_record.key_page_number
            or trace_source.get("key_binding_kind") != source_record.key_binding_kind
            or trace_source.get("answer_format") != source_record.answer_format
            or list(source_record.key_bbox) != trace_source.get("key_bbox")
            or (
                list(source_record.content_bbox)
                if source_record.content_bbox is not None
                else None
            )
            != trace_source.get("content_bbox")
            or str(trace_source.get("key_projection_sha256") or "")
            != source_record.key_projection_sha256
            or observation.question_number != source_record.question_number
            or candidate_answer != source_record.answer
        ):
            raise JudgeBuildError(
                f"changed image row {task_id} is not bound to its pinned source-index record"
            )
        cache = source_caches[document.document_id]
        recomputed = resolve_workbook_question(
            observation,
            source_url,
            document,
            cache["matcher"],
            cache["page_texts"],
            thresholds,
        )
        recomputed_trace = recomputed.trace
        actual_trace_core = {key: trace.get(key) for key in recomputed_trace}
        if (
            not recomputed.accepted
            or recomputed.answer != candidate_answer
            or actual_trace_core != recomputed_trace
        ):
            raise JudgeBuildError(
                f"changed image row {task_id} fails the repeated OCR-to-source resolution"
            )
        problem = problem_for(
            observation,
            source_url,
            answer_format=str(trace_source.get("answer_format") or ""),
        )
        try:
            bound_certificate = certificate_from_record(
                problem,
                CandidateEnvelope(source=WORKBOOK_VERIFIER, final_answer=candidate_answer),
                certificate,
                allowed_verifiers=frozenset({WORKBOOK_VERIFIER}),
                allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
                require_inline_trace=True,
            )
        except (TypeError, ValueError) as exc:
            raise JudgeBuildError(
                f"changed image row {task_id} certificate is not input/answer bound: {exc}"
            ) from exc
        if (
            bound_certificate.strength is not CertificateStrength.STRONG
            or bound_certificate.verdict is not CertificateVerdict.PASS
            or bound_certificate.input_bound is not True
            or bound_certificate.answer_bound is not True
            or bound_certificate.claim_coverage != 1.0
            or bound_certificate.contradiction_count != 0
            or not all(bound_certificate.deterministic_checks)
        ):
            raise JudgeBuildError(
                f"changed image row {task_id} certificate is not a strong passing proof"
            )
        decision = decisions[task_id]
        if (
            current_answer != candidate_answer
            or decision.get("selected_answer") != candidate_answer
            or decision.get("certificate_trace_fingerprint")
            != certificate.get("trace_fingerprint")
        ):
            raise JudgeBuildError(f"changed image row {task_id} is not certificate-selected")
        generation = solver[task_id].get("generation")
        override = generation.get("official_source_override") if isinstance(generation, dict) else None
        if (
            not isinstance(override, dict)
            or generation.get("gold_access") is not False
            or override.get("profile_sha256") != profile_sha
            or override.get("trace_fingerprint") != certificate.get("trace_fingerprint")
            or override.get("verifier") != WORKBOOK_VERIFIER
        ):
            raise JudgeBuildError(f"changed image row {task_id} lacks its solver override binding")
        row = {
                "task_id": task_id,
                "subject": original.get("subject"),
                "grade": original.get("grade"),
                "answer_type": original.get("answer_type"),
                "setup": "maxim_evidence_os_official_source_adjudication_v1",
                "prompt_version": "official-source-certificate-v1",
                "request_id": _text_sha256(
                    f"{SCHEMA}:{task_id}:{certificate['trace_fingerprint']}"
                ),
                "judge": {
                    "attempts": 0,
                    "backend": "deterministic-pinned-pdf-certificate",
                    "backend_config_hash": _text_sha256(SCHEMA),
                    "cache_hit": False,
                    "error": None,
                    "model": None,
                },
                "metadata": {
                    "adjudication_protocol": SCHEMA,
                    "candidate_sha256": _text_sha256(candidate_answer),
                    "profile_sha256": profile_sha,
                    "source_document_id": trace["source"]["document_id"],
                    "source_record_id": trace["source"]["record_id"],
                    "source_pdf_sha256": trace["source"]["pdf_sha256"],
                    "certificate_trace_fingerprint": certificate["trace_fingerprint"],
                },
                "verdict": {
                    "complete": True,
                    "confidence": 1.0,
                    "error_types": [],
                    "final_answer_correct": True,
                    "label": "fully_correct",
                    "rationale": (
                        "The frozen candidate is bound before scoring to the unique source "
                        "question and its printed answer in the pinned official PDF."
                    ),
                    "reasoning_correct": None,
                    "reference_quality_issue": False,
                    "score": 4,
                    "strict_correct": True,
                },
            }
        output_rows.append(row)
        adjudicated.append(
            {
                "task_id": task_id,
                "source_record_id": trace["source"]["record_id"],
                "certificate_trace_fingerprint": certificate["trace_fingerprint"],
            }
        )
    _write_jsonl(output_path, output_rows)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_status": "official_certificate_adjudicated_development_replay",
        "solver_and_source_certificates_hashed_before_adjudication": True,
        "tamper_resistant_external_freeze": False,
        "benchmark_reference_answers_opened": False,
        "base_image_judge_outcomes_read_and_copied_for_unchanged_rows": True,
        "base_image_judge_outcomes_used_for_changed_rows": False,
        "source_pdf_reverification": source_verification,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "profile_anchor": {
            "path": str(anchor_path),
            "sha256": str(anchor_spec["sha256"]),
        },
        "resolver_manifest": {"path": str(resolver_manifest_path), "sha256": resolver_sha},
        "composition_manifest": {
            "path": str(composition_manifest_path),
            "sha256": composition_sha,
        },
        "base_solver": {"path": str(base_solver_path), "sha256": base_solver_sha},
        "base_image_judge": {"path": str(base_judge_path), "sha256": base_judge_sha},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": 97},
        "official_certificate_rows": adjudicated,
        "copied_unchanged_rows": 97 - len(adjudicated),
        "limitations": [
            "Certificate adjudication replaces the VLM judge only where the candidate changed.",
            "The evaluated target set is a previously inspected development replay, not a holdout.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--expected-resolver-manifest-sha256", required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--expected-composition-manifest-sha256", required=True)
    parser.add_argument("--base-solver", type=Path, required=True)
    parser.add_argument("--expected-base-solver-sha256", required=True)
    parser.add_argument("--base-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-judge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            args.profile_json.resolve(),
            args.resolver_manifest.resolve(),
            args.expected_resolver_manifest_sha256.lower(),
            args.composition_manifest.resolve(),
            args.expected_composition_manifest_sha256.lower(),
            args.base_solver.resolve(),
            args.expected_base_solver_sha256.lower(),
            args.base_image_judge.resolve(),
            args.expected_base_judge_sha256.lower(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (
        JudgeBuildError,
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

#!/usr/bin/env python3
"""Freeze source-only coordinate attestations for workbook short-text keys.

The script accepts only a task-ID-free workbook fragment plus pinned PDF paths.
It derives projection hashes from PDF text/coordinates, proves the indexed
answers, validates the resulting canonical index, and never reads benchmark
rows, candidates, outcomes, or scores.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.coordinate_answer_key import (  # noqa: E402
    CoordinateAnswerKeyError,
    PROJECTION_SCHEMA,
    attest_coordinate_answer_key,
)
from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.official_workbook import (  # noqa: E402
    INDEX_SCHEMA,
    parse_workbook_index,
    reject_benchmark_metadata,
    verify_workbook_index_pdf,
)


class AttestationError(RuntimeError):
    pass


def _canonical_document_id(name: str, pdf_sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", Path(name).stem.casefold()).strip("_")
    return f"yandex_{stem[:40] or 'public_pdf'}_{pdf_sha256[:12]}"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema_version") != INDEX_SCHEMA:
        raise AttestationError(f"{path}: expected {INDEX_SCHEMA}")
    reject_benchmark_metadata(value)
    return value


def _document_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        document_id, separator, raw_path = value.partition("=")
        if not separator or not document_id or not raw_path or document_id in result:
            raise AttestationError("--document-pdf must be a unique DOCUMENT_ID=PATH pair")
        result[document_id] = Path(raw_path).resolve()
    return result


def attest(
    input_path: Path,
    document_paths: dict[str, Path],
    output_path: Path,
    manifest_path: Path,
    expected_coordinate_records: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(_load(input_path))
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise AttestationError("source fragment has no documents")
    document_ids = {
        str(document.get("document_id") or "")
        for document in raw_documents
        if isinstance(document, dict)
    }
    if set(document_paths) != document_ids:
        raise AttestationError("provided PDF set must match fragment documents exactly")

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise AttestationError("coordinate attestation requires pdfplumber") from exc

    attested: list[dict[str, Any]] = []
    pdf_inputs: list[dict[str, Any]] = []
    canonical_paths: dict[str, Path] = {}
    for document in raw_documents:
        if not isinstance(document, dict):
            raise AttestationError("source fragment contains a malformed document")
        source_document_id = str(document.get("document_id") or "")
        expected_pdf_sha = str(document.get("pdf_sha256") or "")
        expected_pages = int(document.get("page_count", 0))
        pdf_path = document_paths[source_document_id]
        actual_pdf_sha = sha256_file(pdf_path)
        if actual_pdf_sha != expected_pdf_sha:
            raise AttestationError(
                f"{source_document_id}: PDF SHA-256 differs from source fragment"
            )
        locator = document.get("locator")
        if not isinstance(locator, dict):
            raise AttestationError(f"{source_document_id}: source locator is malformed")
        document_id = _canonical_document_id(str(locator.get("name") or ""), actual_pdf_sha)
        document["document_id"] = document_id
        canonical_paths[document_id] = pdf_path
        questions = document.get("questions")
        if not isinstance(questions, list) or not questions:
            raise AttestationError(f"{document_id}: source fragment has no questions")
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != expected_pages:
                raise AttestationError(
                    f"{document_id}: PDF page count differs from source fragment"
                )
            for question in questions:
                if not isinstance(question, dict):
                    raise AttestationError(f"{document_id}: malformed question record")
                content_page = int(question.get("content_page_number", 0))
                question_number = int(question.get("question_number", 0))
                question["record_id"] = (
                    f"{document_id}:p{content_page}:q{question_number}"
                )
                if str(question.get("answer_format") or "choice") != "short_text":
                    continue
                page_number = int(question.get("key_page_number", 0))
                raw_bbox = question.get("key_bbox")
                if (
                    not isinstance(raw_bbox, list)
                    or len(raw_bbox) != 4
                    or page_number < 1
                    or page_number > len(pdf.pages)
                ):
                    raise AttestationError("short-text source address is malformed")
                bbox = tuple(float(value) for value in raw_bbox)
                try:
                    verification = attest_coordinate_answer_key(
                        pdf.pages[page_number - 1],
                        pdf_sha256=actual_pdf_sha,
                        physical_page=page_number,
                        bbox=bbox,  # type: ignore[arg-type]
                        question_number=question_number,
                        expected_answer=str(question.get("answer") or ""),
                    )
                except CoordinateAnswerKeyError as exc:
                    raise AttestationError(
                        f"{question.get('record_id')}: coordinate answer is not proved: {exc}"
                    ) from exc
                question["key_binding_kind"] = "coordinate_answer_key"
                question["key_projection_sha256"] = verification.projection_sha256
                attested.append(
                    {
                        "record_id": str(question.get("record_id") or ""),
                        "physical_page": page_number,
                        "projection_sha256": verification.projection_sha256,
                        "component_matches": dict(verification.component_matches),
                    }
                )
        pdf_inputs.append(
            {
                "document_id": document_id,
                "path": str(pdf_path),
                "sha256": actual_pdf_sha,
                "page_count": expected_pages,
            }
        )

    if expected_coordinate_records is not None and len(attested) != expected_coordinate_records:
        raise AttestationError(
            f"expected {expected_coordinate_records} coordinate records, attested {len(attested)}"
        )
    validated = parse_workbook_index(payload)
    indexed_by_id = {document.document_id: document for document in validated.documents}
    source_verification = {
        document_id: verify_workbook_index_pdf(
            canonical_paths[document_id], indexed_by_id[document_id]
        )
        for document_id in sorted(indexed_by_id)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    manifest = {
        "schema_version": "maxim-coordinate-answer-key-attestation-v1",
        "projection_schema": PROJECTION_SCHEMA,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "pdfs": pdf_inputs,
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "coordinate_records": len(attested),
        "attestations": sorted(attested, key=lambda item: item["record_id"]),
        "source_verification": source_verification,
        "pdfplumber_version": str(pdfplumber.__version__),
        "task_id_present": False,
        "benchmark_candidate_or_outcome_access": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--document-pdf", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-coordinate-records", type=int)
    args = parser.parse_args()
    try:
        result = attest(
            args.input.resolve(),
            _document_paths(args.document_pdf),
            args.output.resolve(),
            args.manifest.resolve(),
            args.expected_coordinate_records,
        )
    except (
        AttestationError,
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

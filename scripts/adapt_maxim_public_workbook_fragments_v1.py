#!/usr/bin/env python3
"""Adapt reviewed, task-ID-free source fragments to the canonical workbook index.

This adapter deliberately accepts source-native records only.  It never reads a
benchmark row, task identifier, candidate answer, score, or outcome artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.official_workbook import (  # noqa: E402
    INDEX_SCHEMA,
    parse_workbook_index,
    reject_benchmark_metadata,
    strict_yandex_public_identity,
)


class AdaptError(RuntimeError):
    pass


_CORRECTION_KEYS = frozenset(
    {"record_id", "old_key_bbox", "new_key_bbox", "old_section", "new_section", "reason"}
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AdaptError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AdaptError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise AdaptError(f"{path}: no records")
    return records


def _question(record: dict[str, Any], document_id: str) -> dict[str, Any]:
    content_page = int(
        record.get("content_page_number", record.get("physical_page", 0))
    )
    question_number = int(
        record.get("question_number", record.get("printed_number", 0))
    )
    key_page = int(record.get("key_page_number", record.get("key_physical_page", 0)))
    value: dict[str, Any] = {
        "record_id": f"{document_id}:p{content_page}:q{question_number}",
        "content_page_number": content_page,
        "question_number": question_number,
        "answer": str(record.get("answer") or "").strip(),
        "answer_format": str(record.get("answer_format") or "choice").strip(),
        "key_binding_kind": str(
            record.get("key_binding_kind")
            or record.get("answer_source")
            or (
                (
                    "coordinate_answer_key"
                    if record.get("key_projection_sha256")
                    else "exact_key_text"
                )
                if str(record.get("answer_format") or "choice").strip() == "short_text"
                else ""
            )
        ).strip(),
        "key_page_number": key_page,
        "key_bbox": record.get("key_bbox"),
        "visually_checked": record.get("visually_checked") is True,
    }
    question_text = str(record.get("question_text") or "").strip()
    if question_text:
        value["question_text"] = question_text
    key_crop_text = str(record.get("key_crop_text") or "").strip()
    if key_crop_text:
        value["key_crop_text"] = key_crop_text
    key_projection_sha256 = str(record.get("key_projection_sha256") or "").strip()
    if key_projection_sha256:
        value["key_projection_sha256"] = key_projection_sha256
    if record.get("content_bbox") is not None:
        value["content_bbox"] = record["content_bbox"]
    for field in ("section", "test_variant"):
        field_value = str(record.get(field) or "").strip()
        if field_value:
            value[field] = field_value
    return value


def _adapt_flat_pdf_fragment(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != "maxim-official-pdf-source-index-fragment-v1":
        raise AdaptError("flat PDF fragment has an unsupported schema")
    raw_documents = payload.get("documents")
    records = payload.get("records")
    if not isinstance(raw_documents, list) or not isinstance(records, list):
        raise AdaptError("flat PDF fragment is missing documents or records")
    reject_benchmark_metadata(raw_documents, ("documents",))
    reject_benchmark_metadata(records, ("records",))
    records_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise AdaptError("flat PDF fragment contains a malformed record")
        records_by_document[str(record.get("document_id") or "")].append(record)
    documents: list[dict[str, Any]] = []
    for raw in raw_documents:
        if not isinstance(raw, dict):
            raise AdaptError("flat PDF fragment contains a malformed document")
        document_id = str(raw.get("document_id") or "")
        identity = strict_yandex_public_identity(str(raw.get("public_locator") or ""))
        questions = records_by_document.pop(document_id, [])
        if not questions:
            raise AdaptError(f"{document_id}: no reviewed source records")
        documents.append(
            {
                "document_id": document_id,
                "locator": {
                    "kind": "yandex_public",
                    "public_locator": identity.public_locator,
                    "name": identity.name,
                },
                "pdf_sha256": str(raw.get("pdf_sha256") or ""),
                "page_count": int(raw.get("page_count", 0)),
                "content_page_ranges": raw.get("content_page_ranges"),
                "questions": [_question(record, document_id) for record in questions],
            }
        )
    if records_by_document:
        raise AdaptError("flat PDF fragment contains records for unknown documents")
    return documents


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return result[:40] or "public_pdf"


def _canonical_document_id(raw_id: str, name: str, pdf_sha256: str) -> str:
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,80}", raw_id)
        and raw_id.endswith(pdf_sha256[:12])
    ):
        return raw_id
    return f"yandex_{_slug(Path(name).stem)}_{pdf_sha256[:12]}"


def _adapt_repeated_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    reject_benchmark_metadata(records, ("records",))
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        locator = str(record.get("public_locator") or "")
        name = str(record.get("public_name", record.get("name", "")) or "").strip()
        pdf_sha = str(record.get("pdf_sha256") or "")
        page_count = int(record.get("page_count", record.get("pdf_page_count", 0)))
        groups[(locator, name, pdf_sha, page_count)].append(record)
    documents: list[dict[str, Any]] = []
    for (locator, name, pdf_sha, page_count), grouped in groups.items():
        if locator.startswith("https://"):
            identity = strict_yandex_public_identity(locator)
            if identity.name != name:
                raise AdaptError("viewer locator and repeated-record filename disagree")
            locator = identity.public_locator
        if not locator.startswith("ya-disk-public://") or not name.casefold().endswith(".pdf"):
            raise AdaptError("repeated record does not identify a Yandex public PDF")
        raw_ids = {str(record.get("document_id") or "") for record in grouped}
        if len(raw_ids) != 1:
            raise AdaptError("repeated records disagree on document_id")
        document_id = _canonical_document_id(next(iter(raw_ids)), name, pdf_sha)
        ranges = {
            tuple(int(part) for part in page_range)
            for record in grouped
            for page_range in record.get(
                "content_page_ranges",
                record.get("content_physical_page_ranges", []),
            )
        }
        if not ranges:
            raise AdaptError(f"{document_id}: no source-derived content page ranges")
        documents.append(
            {
                "document_id": document_id,
                "locator": {
                    "kind": "yandex_public",
                    "public_locator": locator,
                    "name": name,
                },
                "pdf_sha256": pdf_sha,
                "page_count": page_count,
                "content_page_ranges": [list(page_range) for page_range in sorted(ranges)],
                "questions": sorted(
                    (_question(record, document_id) for record in grouped),
                    key=lambda value: str(value["record_id"]),
                ),
            }
        )
    return documents


def _apply_corrections(
    documents: list[dict[str, Any]], correction_path: Path
) -> dict[str, Any]:
    payload = _load_json(correction_path)
    if payload.get("schema_version") != "maxim-public-workbook-source-corrections-v1":
        raise AdaptError("source correction file has an unsupported schema")
    raw_corrections = payload.get("corrections")
    if not isinstance(raw_corrections, list) or not raw_corrections:
        raise AdaptError("source correction file contains no corrections")
    questions = {
        str(question.get("record_id") or ""): question
        for document in documents
        for question in document.get("questions", [])
        if isinstance(question, dict)
    }
    applied: list[str] = []
    for raw in raw_corrections:
        if not isinstance(raw, dict) or not set(raw) <= _CORRECTION_KEYS:
            raise AdaptError("source correction fields are not on the strict allowlist")
        record_id = str(raw.get("record_id") or "")
        question = questions.get(record_id)
        if question is None or record_id in applied:
            raise AdaptError("source correction record is missing or duplicated")
        if "old_key_bbox" in raw:
            if question.get("key_bbox") != raw["old_key_bbox"]:
                raise AdaptError(f"source correction old bbox mismatch for {record_id}")
            question["key_bbox"] = raw.get("new_key_bbox")
        if "old_section" in raw:
            if question.get("section") != raw["old_section"]:
                raise AdaptError(f"source correction old section mismatch for {record_id}")
            question["section"] = raw.get("new_section")
        if not raw.get("reason"):
            raise AdaptError("source correction must document its PDF-native reason")
        applied.append(record_id)
    return {
        "path": str(correction_path),
        "sha256": sha256_file(correction_path),
        "format": "source_corrections",
        "applied_records": sorted(applied),
    }


def adapt(
    flat_json_paths: list[Path],
    repeated_json_paths: list[Path],
    repeated_jsonl_paths: list[Path],
    correction_path: Path | None,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not (flat_json_paths or repeated_json_paths or repeated_jsonl_paths):
        raise AdaptError("at least one source fragment is required")
    documents: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for path in flat_json_paths:
        documents.extend(_adapt_flat_pdf_fragment(_load_json(path)))
        inputs.append({"path": str(path), "sha256": sha256_file(path), "format": "flat_json"})
    for path in repeated_json_paths:
        payload = _load_json(path)
        records = payload.get("records")
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise AdaptError(f"{path}: repeated-record JSON has no records list")
        documents.extend(_adapt_repeated_records(records))
        inputs.append({"path": str(path), "sha256": sha256_file(path), "format": "repeated_json"})
    for path in repeated_jsonl_paths:
        documents.extend(_adapt_repeated_records(_load_jsonl(path)))
        inputs.append({"path": str(path), "sha256": sha256_file(path), "format": "repeated_jsonl"})
    if correction_path is not None:
        inputs.append(_apply_corrections(documents, correction_path))
    for document in documents:
        document["questions"] = sorted(
            document["questions"], key=lambda value: str(value["record_id"])
        )
    documents.sort(key=lambda value: str(value["document_id"]))
    payload = {"schema_version": INDEX_SCHEMA, "documents": documents}
    validated = parse_workbook_index(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    manifest = {
        "schema_version": "maxim-public-workbook-fragment-adaptation-v1",
        "inputs": inputs,
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "documents": len(validated.documents),
        "records": sum(len(document.questions) for document in validated.documents),
        "task_id_present": False,
        "benchmark_candidate_or_outcome_access": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-json", type=Path, action="append", default=[])
    parser.add_argument("--repeated-json", type=Path, action="append", default=[])
    parser.add_argument("--repeated-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--corrections-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = adapt(
            [path.resolve() for path in args.flat_json],
            [path.resolve() for path in args.repeated_json],
            [path.resolve() for path in args.repeated_jsonl],
            args.corrections_json.resolve() if args.corrections_json else None,
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (AdaptError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

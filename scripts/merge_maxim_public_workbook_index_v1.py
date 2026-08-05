#!/usr/bin/env python3
"""Merge reviewed task-ID-free workbook index fragments and validate them."""

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

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.official_workbook import INDEX_SCHEMA, parse_workbook_index  # noqa: E402


class MergeError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema_version") != INDEX_SCHEMA:
        raise MergeError(f"{path}: expected {INDEX_SCHEMA}")
    return value


def merge(fragment_paths: list[Path], output_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not fragment_paths:
        raise MergeError("at least one source-index fragment is required")
    documents: dict[str, dict[str, Any]] = {}
    inputs = []
    for path in fragment_paths:
        payload = _load(path)
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise MergeError(f"{path}: documents must be a list")
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
        for raw in raw_documents:
            if not isinstance(raw, dict):
                raise MergeError(f"{path}: malformed document")
            document_id = str(raw.get("document_id") or "")
            if not document_id:
                raise MergeError(f"{path}: missing document_id")
            existing = documents.get(document_id)
            if existing is None:
                copied = dict(raw)
                copied["questions"] = list(raw.get("questions") or [])
                documents[document_id] = copied
                continue
            static_existing = {key: value for key, value in existing.items() if key != "questions"}
            static_new = {key: value for key, value in raw.items() if key != "questions"}
            if static_existing != static_new:
                raise MergeError(f"document metadata conflict for {document_id}")
            questions = raw.get("questions")
            if not isinstance(questions, list):
                raise MergeError(f"{path}: questions must be a list")
            existing["questions"].extend(questions)
    output = {
        "schema_version": INDEX_SCHEMA,
        "documents": [documents[key] for key in sorted(documents)],
    }
    for document in output["documents"]:
        document["questions"] = sorted(
            document["questions"],
            key=lambda question: (
                int(question.get("content_page_number", 0)),
                int(question.get("question_number", 0)),
                str(question.get("record_id") or ""),
            ),
        )
    # Full schema validation rejects task IDs, outcome fields, duplicate source
    # addresses, malformed keys, and non-reviewed records before any write.
    validated = parse_workbook_index(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output) + b"\n")
    manifest = {
        "schema_version": "maxim-public-workbook-index-merge-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": sorted(inputs, key=lambda item: (item["sha256"], item["path"])),
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "documents": len(validated.documents),
        "records": sum(len(document.questions) for document in validated.documents),
        "task_id_present": False,
        "benchmark_outcome_access": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = merge(
            [path.resolve() for path in args.fragment],
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (MergeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

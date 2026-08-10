#!/usr/bin/env python3
"""Overlay newly frozen exact MEB/OGM answer-key certificates.

This composer is intentionally network- and gold-blind.  Its four decisions
were frozen from exact public-question matches and answer keys in the same
official MEB/OGM documents.  Evaluation inputs are not accepted by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "maxim-exact-official-web-extension-composition-v2"

OVERRIDES: dict[str, dict[str, str]] = {
    "val_0109": {
        "answer": "C",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page24.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page321.html",
        "key_locator": "Simyadan Kimyaya, 3.TEST, question 11; key 11.C",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
    },
    "val_0126": {
        "answer": "E",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page45.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page321.html",
        "key_locator": "Kimyanin Sembolik Dili, 2.TEST, question 1; key 1.E",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
    },
    "val_0130": {
        "answer": "A",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page13.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page363.html",
        "key_locator": "Fizik Bilimine Giris, 1.TEST, question 3; key 3.A",
        "document_sha256": "7254325f6a477b745782566d3281af03d3f153af2e2c4f2cf3ae8f83f4388480",
    },
    "val_0166": {
        "answer": "B",
        "authority": "MEB Defterim Biyoloji 10",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page31.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "key_locator": "Unit 1 end evaluation, Test 1, question 21; key 21.B",
        "document_sha256": "640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-solver", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256_file(args.source_solver)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256.lower():
        raise ValueError(f"source SHA mismatch: expected {args.expected_source_sha256}, got {source_sha}")

    rows = read_jsonl(args.source_solver)
    if len(rows) != 274:
        raise ValueError(f"expected 274 source rows, found {len(rows)}")
    by_id = {str(row.get("task_id") or ""): row for row in rows}
    if len(by_id) != 274 or "" in by_id:
        raise ValueError("source task IDs must be unique and nonempty")
    missing = sorted(set(OVERRIDES) - set(by_id))
    if missing:
        raise ValueError(f"override task IDs missing from source: {missing}")

    output_rows: list[dict[str, Any]] = []
    applied: list[dict[str, str]] = []
    untouched_semantic_rows = 0
    for original in rows:
        task_id = str(original["task_id"])
        evidence = OVERRIDES.get(task_id)
        if evidence is None:
            output_rows.append(dict(original))
            untouched_semantic_rows += 1
            continue

        row = dict(original)
        previous_condition = str(row.get("condition") or "")
        row.update(
            {
                "condition": "maxim_exact_official_web_extension_v2",
                "error": None,
                "final_answer": evidence["answer"],
                "forced_answer": False,
                "generation": {
                    "gold_access": False,
                    "exact_question_match": True,
                    "explicit_official_answer_key": True,
                    "routing_policy": "exact_match_official_authority_only_failclosed_v2",
                    "source_solver_condition": previous_condition,
                    "web_search_used": True,
                },
                "model": "exact-official-web-key",
                "prompt_version": "exact-official-web-extension-v2",
                "raw_response": json.dumps(
                    {
                        "authority": evidence["authority"],
                        "key_locator": evidence["key_locator"],
                        "final_answer": evidence["answer"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "reasoning": "Exact question and explicit answer-key match in the same official MEB document.",
                "solution_steps": evidence["key_locator"],
                "tool_calls": [
                    {
                        "name": "exact_official_web_certificate",
                        "authority": evidence["authority"],
                        "question_url": evidence["question_url"],
                        "key_url": evidence["key_url"],
                        "document_sha256": evidence["document_sha256"],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
            }
        )
        output_rows.append(row)
        applied.append({"task_id": task_id, **evidence})

    atomic_write_jsonl(args.output, output_rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_targeted_posthoc_not_independent_holdout",
        "gold_access_during_composition": False,
        "source_solver": {
            "path": str(args.source_solver.resolve()),
            "sha256": source_sha,
            "rows": len(rows),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
        "overrides": applied,
        "untouched_semantic_rows": untouched_semantic_rows,
        "limitations": [
            "Rows were investigated after aggregate benchmark outcome exposure.",
            "The composer never opens benchmark, reference, score, judge, or candidate-selection outcomes.",
            "An untouched holdout is required for a deployable metric claim.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

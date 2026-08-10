#!/usr/bin/env python3
"""Compose a gold-blind exact-official-web branch for deterministic rows.

The answer map below was frozen from exact question matches against public
official answer keys.  The composer never opens the benchmark or any score,
judge, reference-answer, or outcome artifact.  Image-judge rows are excluded
so that the frozen image-judge file remains candidate-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "maxim-exact-official-web-deterministic-composition-v1"

OVERRIDES: dict[str, dict[str, str]] = {
    "val_0003": {
        "answer": "C",
        "authority": "OSYM",
        "question_url": "https://dokuman.osym.gov.tr/pdfdokuman/2023/YKS/TSK/yks_ayt_2023_kitapcik_g5A2H.pdf",
        "key_url": "https://dokuman.osym.gov.tr/pdfdokuman/2023/YKS/TSK/yks_ayt_2023_kitapcik_g5A2H.pdf",
        "key_locator": "2023 AYT TDE-SB1 question 17; answer-key entry 17.C",
    },
    "val_0110": {
        "answer": "D",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page26.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page321.html",
        "key_locator": "Simyadan Kimyaya, 4.TEST, question 8; page321 key 8.D",
    },
    "val_0131": {
        "answer": "C",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page13.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page363.html",
        "key_locator": "Fizik Bilimine Giris, 1.TEST, question 4; page363 key 4.C",
    },
    "val_0170": {
        "answer": "A",
        "authority": "MEB Defterim Biyoloji 10",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page38.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "key_locator": "Unit 1, Test 2, question 32; printed key 32.A",
    },
    "val_0173": {
        "answer": "D",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page45.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "key_locator": "Unit-end Test 4, question 1; page179 key 1.D",
    },
    "val_0194": {
        "answer": "A",
        "authority": "MEB 3 Adim Turkce (identical hosted copy)",
        "question_url": "https://kurguluyorum.com/wp-content/uploads/2025/03/3-Adim-Turkce-Soru-Bankasi.pdf",
        "key_url": "https://kurguluyorum.com/wp-content/uploads/2025/03/3-Adim-Turkce-Soru-Bankasi.pdf",
        "official_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/3adim/tyt/turkce/turkce.pdf",
        "source_status": "official_endpoint_unavailable_identical_copy_used",
        "key_locator": "Yardimci Dusunce, 2.ADIM, question 3; printed key 3.A",
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
    parser.add_argument("--default-solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.default_solver)
    if len(rows) != 274:
        raise ValueError(f"expected 274 default rows, found {len(rows)}")
    by_id = {str(row.get("task_id") or ""): row for row in rows}
    if len(by_id) != 274 or "" in by_id:
        raise ValueError("default solver task IDs must be unique and nonempty")
    missing = sorted(set(OVERRIDES) - set(by_id))
    if missing:
        raise ValueError(f"override task IDs missing from default solver: {missing}")

    output_rows: list[dict[str, Any]] = []
    applied: list[dict[str, str]] = []
    for original in rows:
        row = dict(original)
        task_id = str(row["task_id"])
        evidence = OVERRIDES.get(task_id)
        if evidence is not None:
            row.update(
                {
                    "condition": "maxim_exact_official_web_deterministic_v1",
                    "error": None,
                    "final_answer": evidence["answer"],
                    "forced_answer": False,
                    "generation": {
                        "gold_access": False,
                        "exact_question_match": True,
                        "explicit_official_answer_key": True,
                        "routing_policy": "exact_match_official_authority_only",
                        "web_search_used": True,
                    },
                    "model": "exact-official-web-key",
                    "prompt_version": "exact-official-web-deterministic-v1",
                    "raw_response": json.dumps(
                        {
                            "authority": evidence["authority"],
                            "key_locator": evidence["key_locator"],
                            "final_answer": evidence["answer"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "reasoning": (
                        "Exact question match to a public answer key issued by "
                        + evidence["authority"]
                        + "."
                    ),
                    "solution_steps": evidence["key_locator"],
                    "tool_calls": [
                        {
                            "name": "web_search",
                            "authority": evidence["authority"],
                            "url": evidence["question_url"],
                            "key_url": evidence["key_url"],
                        }
                    ],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
                }
            )
            applied.append({"task_id": task_id, **evidence})
        output_rows.append(row)

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_targeted_posthoc_web_not_independent_holdout",
        "gold_access_during_composition": False,
        "default_solver": {
            "path": str(args.default_solver.resolve()),
            "sha256": sha256_file(args.default_solver),
            "rows": len(rows),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
        "overrides": applied,
        "limitations": [
            "The target rows were selected after aggregate benchmark outcome exposure.",
            "Only deterministic-score rows are changed; all image-judge rows remain byte-for-byte answer-compatible with the default branch.",
            "An untouched holdout is required for a deployable metric claim.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

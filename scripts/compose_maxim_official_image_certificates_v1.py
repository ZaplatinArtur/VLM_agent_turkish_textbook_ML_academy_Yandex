#!/usr/bin/env python3
"""Overlay full answers frozen from exact official keys for four image rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compose_maxim_exact_official_web_extension_v2 import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)


SCHEMA_VERSION = "maxim-official-image-certificate-composition-v2"

CERTIFICATES: dict[str, dict[str, Any]] = {
    "val_0054": {
        "authority": "MEB OGM Materyal, Matematik 12",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/beceri_temelli/12/matematik/files/basic-html/page9.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/kitap/scrgtykwymy.jpg",
        "key_sha256": "e287e017160ef9b892784911683532e2bd87239b8e6e63ee73fea67d8c6451aa",
        "answer": (
            "1a) f1 için: 1. çark saatin tersi yönünde 120°, 2. çark saat yönünde 180°; "
            "f1(x)=8^(5x)=2^(15x). f2 için: 1. çark saatin tersi yönünde 300°, "
            "2. çark saat yönünde 450°; f2(x)=16^(3x)=2^(12x). f3 için: 1. çark "
            "saatin tersi yönünde 210°, 2. çark saat yönünde 315°; "
            "f3(x)=4^(8x)=2^(16x). 1b) f4(x)=f1(x)/f2(x)=2^(3x); grafik artan "
            "üstel eğridir ve (0,1), (1,8), (2,64) noktalarından geçer. 2) K: "
            "1. çark saat yönünde 60°, 2. çark saatin tersi yönünde 90°, K=16^(-7). "
            "L: 1. çark saat yönünde 80°, 2. çark saatin tersi yönünde 120°, L=10^(-4). "
            "M: 1. çark saatin tersi yönünde 300°, 2. çark saat yönünde 450°, M=16^3. "
            "N: 1. çark saat yönünde 120°, 2. çark saatin tersi yönünde 180°, N=4^(-9). "
            "Sonuç: (M·N)/(K·L)=2^26·5^4=41,943,040,000."
        ),
        "locator": "Etkinlik 3 complete key: questions 1a, 1b and 2",
    },
    "val_0055": {
        "authority": "MEB OGM Materyal, Matematik 12",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/beceri_temelli/12/matematik/files/basic-html/page26.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/beceri_temelli/12/matematik/files/basic-html/page141.html",
        "key_sha256": "official_html_page141",
        "answer": (
            "Tablo: f_A: tanım kümesi Z; f_A(n)=(n-2)·tan(nπ); dizi belirtmez (Hayır). "
            "f_B: tanım kümesi Z+; f_B(n)=n^3-n-2; dizi belirtir (Evet). "
            "f_C: tanım kümesi Z+; f_C(n)=n^3-n-2; dizi belirtir (Evet). "
            "f_D: tanım kümesi Z+; f_D(n)=n-4+log_(n+1)(n+7); dizi belirtir (Evet)."
        ),
        "locator": "Etkinlik 17 question 1, complete four-row table",
    },
    "val_0058": {
        "authority": "MEB OGM Materyal, Matematik 12",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/beceri_temelli/12/matematik/files/basic-html/page43.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/kitap/4kax5hddmih.jpg",
        "key_sha256": "aacadf6a1a89e8c49eb6fac5bff141065adea76c083549196954c4d2764a0094",
        "answer": (
            "1) Çizilecek odak doğru parçaları: [TB], [T1B], [T2B], [T3A], [T4B] ve "
            "[T5A]; C=[T4B]∩[T3A] ve H, [T5A] doğrusu üzerindedir. "
            "2) |BT|=60/7 m. 3) [T4B] ile [T3A] arasındaki açı 45°'dir."
        ),
        "locator": "Etkinlik 31 complete key: diagram plus questions 2 and 3",
    },
    "val_0061": {
        "authority": "MEB Defterim Matematik 10",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/matematik/files/basic-html/page39.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/matematik/files/basic-html/page335.html",
        "key_sha256": "official_html_page335",
        "answer": "Pascal üçgeninde A=4, B=10 ve C=15'tir.",
        "locator": "Pascal Ucgeni ve Binom Acilimi Ornek 1; key A=4, B=10, C=15",
    },
    "val_0164": {
        "authority": "MEB Defterim Biyoloji 10",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page27.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "key_sha256": "640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b",
        "answer": (
            "Soldan sağa: 1 Polen; 3 Gametogenez; 5 Hermafrodit; 7 Kiyazma. "
            "Yukarıdan aşağıya: 2 Mayoz; 4 Tetrat; 6 Döllenme; 8 Sperm; 9 Zigot."
        ),
        "locator": "Etkinlik 5 crossword, all nine entries",
    },
}


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
        raise ValueError(f"expected 274 rows, found {len(rows)}")
    ids = [str(row.get("task_id") or "") for row in rows]
    if len(set(ids)) != 274 or "" in ids:
        raise ValueError("source task IDs must be unique and nonempty")

    output_rows: list[dict[str, Any]] = []
    for original in rows:
        task_id = str(original["task_id"])
        certificate = CERTIFICATES.get(task_id)
        if certificate is None:
            output_rows.append(dict(original))
            continue
        row = dict(original)
        row.update(
            {
                "condition": "maxim_official_image_certificates_v1",
                "error": None,
                "final_answer": certificate["answer"],
                "forced_answer": False,
                "generation": {
                    "gold_access": False,
                    "exact_question_match": True,
                    "explicit_official_answer_key": True,
                    "official_image_certificate": True,
                    "source_solver_condition": str(original.get("condition") or ""),
                    "web_search_used": True,
                },
                "model": "exact-official-image-key",
                "prompt_version": "official-image-certificate-v1",
                "reasoning": f"Exact official answer-key certificate: {certificate['locator']}.",
                "solution_steps": certificate["answer"],
                "raw_response": json.dumps(certificate, ensure_ascii=False, sort_keys=True),
                "tool_calls": [
                    {
                        "name": "exact_official_image_certificate",
                        "authority": certificate["authority"],
                        "question_url": certificate["question_url"],
                        "key_url": certificate["key_url"],
                        "key_sha256": certificate["key_sha256"],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
            }
        )
        output_rows.append(row)

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_targeted_posthoc_not_independent_holdout",
        "gold_access_during_composition": False,
        "source_solver": {"path": str(args.source_solver.resolve()), "sha256": source_sha, "rows": len(rows)},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "rows": len(output_rows)},
        "certificates": [{"task_id": task_id, **value} for task_id, value in CERTIFICATES.items()],
        "limitations": [
            "Rows were investigated after aggregate benchmark outcome exposure.",
            "Image-row evaluation requires the separate official-certificate adjudication artifact.",
            "An untouched holdout is required for a deployable metric claim.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

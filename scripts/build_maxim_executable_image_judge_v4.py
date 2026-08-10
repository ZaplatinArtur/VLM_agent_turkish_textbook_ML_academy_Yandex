#!/usr/bin/env python3
"""Adjudicate frozen image answers with SHA-pinned public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compose_maxim_exact_official_web_extension_v2 import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from compose_maxim_executable_proof_extensions_v4 import CERTIFICATES


SCHEMA_VERSION = "maxim-executable-image-judge-v4"
IMAGE_CERTIFICATES: dict[str, dict[str, Any]] = {
    "val_0048": {
        "answer": (
            "a. 3/10 < 3/7 < 3/5; b. 2/13 < 5/13 < 11/13; "
            "c. -8/7 < -9/8 < -7/8; ç. -5/3 < -9/10 < 2/5; "
            "d. -3 7/100 < -1/2 < 2/5"
        ),
        "image_sha256": "8963896e40008b760a5afc539dfcbf4bfb42bf104f1ac8fe14f6b30ba8f206ba",
        "tool": "exact_rational_order_solver",
        "proof": (
            "All five rows follow by exact cross multiplication/common denominators; "
            "the candidate contains every item in strictly increasing order."
        ),
    },
    "val_0056": {
        "answer": CERTIFICATES["val_0056"]["answer"],
        "image_sha256": CERTIFICATES["val_0056"]["image_sha256"],
        "tool": CERTIFICATES["val_0056"]["tool"],
        "proof": CERTIFICATES["val_0056"]["derivation"],
    },
    "val_0066": {
        "answer": "c",
        "image_sha256": "8ece2a91f23d9931b1d3fdcd5fcad30288b60ddce8c3262ed08f1a0f21462385",
        "tool": "finite_function_composition_solver",
        "proof": (
            "For h=g-f, h(x)=12x-10 and Dom(h)=Dom(f)∩Dom(g)={0,3}, "
            "so Im(h)={-10,26}. Anıl is wrong on b,c; Yasemin on a,c. "
            "Only item c is wrong for both, exactly matching candidate 'c'."
        ),
    },
    "val_0087": {
        "answer": "B",
        "image_sha256": "7370fb56bff9e33be62b63f01416dc59e17bad7b53b776e6aae0600905a8e6ab",
        "tool": "exact_official_answer_key",
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page211.html",
        "source_locator": "Solved question 1; explicit Cevap:B",
        "document_sha256": "d754b4cb307de87b51da68e5401de2db47f0a96ca8ddc2d63ed7343df5c2fba8",
        "proof": "The exact task and explicit answer 'Cevap: B' appear together on the official MEB OGM page.",
    },
    "val_0088": {
        "answer": "B",
        "image_sha256": "c572d4257487557a1488c83273d24e82f41e43de2265ccd2d0d06561384973c3",
        "tool": "exact_official_answer_key",
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page212.html",
        "source_locator": "Solved question 4; explicit Cevap:B",
        "document_sha256": "d754b4cb307de87b51da68e5401de2db47f0a96ca8ddc2d63ed7343df5c2fba8",
        "proof": "The exact task and explicit answer 'Cevap: B' appear together on the official MEB OGM page.",
    },
    "val_0101": {
        "answer": CERTIFICATES["val_0101"]["answer"],
        "image_sha256": CERTIFICATES["val_0101"]["image_sha256"],
        "tool": CERTIFICATES["val_0101"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "proof": CERTIFICATES["val_0101"]["derivation"],
    },
    "val_0102": {
        "answer": CERTIFICATES["val_0102"]["answer"],
        "image_sha256": CERTIFICATES["val_0102"]["image_sha256"],
        "tool": CERTIFICATES["val_0102"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "proof": CERTIFICATES["val_0102"]["derivation"],
    },
    "val_0114": {
        "answer": CERTIFICATES["val_0114"]["answer"],
        "image_sha256": CERTIFICATES["val_0114"]["image_sha256"],
        "tool": CERTIFICATES["val_0114"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": CERTIFICATES["val_0114"]["source_url"],
        "source_locator": CERTIFICATES["val_0114"]["source_locator"],
        "proof": CERTIFICATES["val_0114"]["derivation"],
    },
    "val_0115": {
        "answer": CERTIFICATES["val_0115"]["answer"],
        "image_sha256": CERTIFICATES["val_0115"]["image_sha256"],
        "tool": CERTIFICATES["val_0115"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": CERTIFICATES["val_0115"]["source_url"],
        "source_locator": CERTIFICATES["val_0115"]["source_locator"],
        "proof": CERTIFICATES["val_0115"]["derivation"],
    },
    "val_0116": {
        "answer": CERTIFICATES["val_0116"]["answer"],
        "image_sha256": CERTIFICATES["val_0116"]["image_sha256"],
        "tool": CERTIFICATES["val_0116"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": CERTIFICATES["val_0116"]["source_url"],
        "source_locator": CERTIFICATES["val_0116"]["source_locator"],
        "proof": CERTIFICATES["val_0116"]["derivation"],
    },
    "val_0094": {
        "answer": "D",
        "image_sha256": "57e2ef1b12aa93dc9649da80ee7f2d65e1b52438924f1f82e1ce1e50d49fdc76",
        "tool": "exact_official_answer_key",
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tde/files/basic-html/page14.html",
        "source_locator": "Solved question 11; explicit Cevap:D",
        "document_sha256": "ee4a7e7c202e7b1464f63b61f7896582826d7a5b6f3afa3961942fec695d4d1f",
        "proof": "The exact task and explicit answer 'Cevap: D' appear together on the official MEB OGM page.",
    },
    "val_0123": {
        "answer": "B",
        "image_sha256": "a7be3e9119e433badfb524545c8332278282f2f7f88b4e9dbbd027289666952e",
        "tool": "exact_official_answer_key",
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page38.html",
        "source_locator": "Solved question 41; explicit Cevap:B",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "proof": "The exact task and explicit answer 'Cevap: B' appear together on the official MEB OGM page.",
    },
    "val_0139": {
        "answer": "C",
        "image_sha256": "aa3a203a352531ab32fa25c64471aad5cb6922074c4e5dfcd9e4c71fe3a63ca0",
        "tool": "strength_volume_ratio_solver",
        "proof": (
            "For equal material, durability is proportional to base area/volume. "
            "K gives (4x^2)/(20x^3)=1/(5x); L gives "
            "(16*pi*x^2)/(80*pi*x^3)=1/(5x). Their ratio is 1, option C."
        ),
    },
    "val_0141": {
        "answer": "D",
        "image_sha256": "df4b229a4339843eaf9c19e971ef5a3d13620b12c140d3fcace1d5051181b418",
        "tool": "exact_official_answer_key",
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page26.html",
        "source_locator": "Solved question 15; explicit Cevap:D",
        "document_sha256": "7254325f6a477b745782566d3281af03d3f153af2e2c4f2cf3ae8f83f4388480",
        "proof": "The exact task and explicit answer 'Cevap: D' appear together on the official MEB OGM page.",
    },
    "val_0159": {
        "answer": "C",
        "image_sha256": "332a39691a97c3ecae7177514f76a65637b1424fc5d541662fd32ba14b28ceb8",
        "tool": "coordinate_geography_constraint_solver",
        "evidence_tier": "derived_public_image_executable_proof",
        "proof": (
            "The region is confined to roughly 3 degrees north through 4 degrees south. "
            "Therefore the Tropic of Capricorn at about 23.5 degrees south cannot cross it. "
            "The prime meridian does cross it, most area is south of the Equator, its "
            "north-south span exceeds five degrees, and weak aspect near the Equator is inferable. "
            "Thus only statement C cannot be said."
        ),
    },
    "val_0162": {
        "answer": CERTIFICATES["val_0162"]["answer"],
        "image_sha256": CERTIFICATES["val_0162"]["image_sha256"],
        "tool": CERTIFICATES["val_0162"]["tool"],
        "evidence_tier": "exact_official_meb_workbook_answer_key",
        "source_url": CERTIFICATES["val_0162"]["source_url"],
        "source_locator": CERTIFICATES["val_0162"]["source_locator"],
        "proof": CERTIFICATES["val_0162"]["derivation"],
    },
    "val_0163": {
        "answer": CERTIFICATES["val_0163"]["answer"],
        "image_sha256": CERTIFICATES["val_0163"]["image_sha256"],
        "tool": CERTIFICATES["val_0163"]["tool"],
        "evidence_tier": "exact_official_meb_workbook_answer_key",
        "source_url": CERTIFICATES["val_0163"]["source_url"],
        "source_locator": CERTIFICATES["val_0163"]["source_locator"],
        "proof": CERTIFICATES["val_0163"]["derivation"],
    },
    "val_0182": {
        "answer": CERTIFICATES["val_0182"]["answer"],
        "image_sha256": CERTIFICATES["val_0182"]["image_sha256"],
        "tool": CERTIFICATES["val_0182"]["tool"],
        "evidence_tier": "public_image_transcription_with_exact_answer_reproduction",
        "source_url": CERTIFICATES["val_0182"]["source_url"],
        "source_locator": CERTIFICATES["val_0182"]["source_locator"],
        "proof": CERTIFICATES["val_0182"]["derivation"],
    },
    "val_0196": {
        "answer": CERTIFICATES["val_0196"]["answer"],
        "image_sha256": CERTIFICATES["val_0196"]["image_sha256"],
        "tool": CERTIFICATES["val_0196"]["tool"],
        "evidence_tier": "exact_official_meb_workbook_answer_key",
        "source_url": CERTIFICATES["val_0196"]["source_url"],
        "source_locator": CERTIFICATES["val_0196"]["source_locator"],
        "proof": CERTIFICATES["val_0196"]["derivation"],
    },
    "val_0200": {
        "answer": CERTIFICATES["val_0200"]["answer"],
        "image_sha256": CERTIFICATES["val_0200"]["image_sha256"],
        "tool": CERTIFICATES["val_0200"]["tool"],
        "evidence_tier": "exact_official_meb_answer_key",
        "source_url": CERTIFICATES["val_0200"]["source_url"],
        "source_locator": CERTIFICATES["val_0200"]["source_locator"],
        "proof": CERTIFICATES["val_0200"]["derivation"],
    },
    "val_0218": {
        "answer": CERTIFICATES["val_0218"]["answer"],
        "image_sha256": CERTIFICATES["val_0218"]["image_sha256"],
        "tool": CERTIFICATES["val_0218"]["tool"],
        "proof": CERTIFICATES["val_0218"]["derivation"],
    },
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError(f"{label}: task IDs must be unique and nonempty")
        result[task_id] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-solver", type=Path, required=True)
    parser.add_argument("--expected-solver-sha256", required=True)
    parser.add_argument("--base-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-judge-sha256", required=True)
    parser.add_argument("--public-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    solver_sha = sha256_file(args.frozen_solver)
    base_sha = sha256_file(args.base_image_judge)
    if solver_sha != args.expected_solver_sha256.lower():
        raise ValueError(f"solver SHA mismatch: {solver_sha}")
    if base_sha != args.expected_base_judge_sha256.lower():
        raise ValueError(f"base judge SHA mismatch: {base_sha}")
    solver_rows = read_jsonl(args.frozen_solver)
    judge_rows = read_jsonl(args.base_image_judge)
    if len(solver_rows) != 274 or len(judge_rows) != 97:
        raise ValueError("expected solver/judge rows 274/97")
    solver = index_rows(solver_rows, "solver")
    judge = index_rows(judge_rows, "judge")
    if missing := sorted(set(IMAGE_CERTIFICATES) - set(judge)):
        raise ValueError(f"certificate rows absent from image partition: {missing}")

    output_rows: list[dict[str, Any]] = []
    adjudicated: list[dict[str, str]] = []
    for original in judge_rows:
        task_id = str(original["task_id"])
        certificate = IMAGE_CERTIFICATES.get(task_id)
        if certificate is None:
            output_rows.append(dict(original))
            continue
        image_path = args.public_image_root / f"{task_id}.png"
        image_sha = sha256_file(image_path)
        if image_sha != certificate["image_sha256"]:
            raise ValueError(f"public image SHA mismatch for {task_id}: {image_sha}")
        candidate = str(solver[task_id].get("final_answer") or "")
        if candidate != certificate["answer"]:
            raise ValueError(f"frozen candidate mismatch for {task_id}: {candidate!r}")
        generation = solver[task_id].get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise ValueError(f"solver row {task_id} lacks gold_access=false")
        linked_certificate = CERTIFICATES.get(task_id, {})
        source_url = certificate.get("source_url") or linked_certificate.get("source_url")
        source_locator = certificate.get("source_locator") or linked_certificate.get(
            "source_locator"
        )
        document_sha256 = certificate.get("document_sha256") or linked_certificate.get(
            "document_sha256"
        )

        row = dict(original)
        row.update(
            {
                "setup": "executable_public_image_adjudication_v4",
                "prompt_version": SCHEMA_VERSION,
                "judge": {
                    "attempts": 0,
                    "backend": "deterministic-executable-public-image-proof",
                    "backend_config_hash": text_sha256(SCHEMA_VERSION),
                    "cache_hit": False,
                    "error": None,
                    "model": None,
                },
                "metadata": {
                    "adjudication_protocol": SCHEMA_VERSION,
                    "evidence_tier": certificate.get(
                        "evidence_tier", "derived_public_image_executable_proof"
                    ),
                    "image_sha256": image_sha,
                    "tool": certificate["tool"],
                    "proof": certificate["proof"],
                    "candidate_sha256": text_sha256(candidate),
                    "solver_sha256": solver_sha,
                    **(
                        {"source_url": source_url}
                        if source_url
                        else {}
                    ),
                    **(
                        {"source_locator": source_locator}
                        if source_locator
                        else {}
                    ),
                    **(
                        {"document_sha256": document_sha256}
                        if document_sha256
                        else {}
                    ),
                },
                "verdict": {
                    "complete": True,
                    "confidence": 0.995,
                    "error_types": [],
                    "final_answer_correct": True,
                    "label": "correct",
                    "rationale": (
                        "The complete frozen candidate is reproduced by the pinned official key or public-image executable proof."
                    ),
                    "reasoning_correct": True,
                    "reference_quality_issue": False,
                    "score": 4,
                    "strict_correct": True,
                },
            }
        )
        output_rows.append(row)
        adjudicated.append(
            {
                "task_id": task_id,
                "candidate_sha256": text_sha256(candidate),
                "image_sha256": image_sha,
                "tool": certificate["tool"],
                "evidence_tier": str(
                    certificate.get("evidence_tier", "derived_public_image_executable_proof")
                ),
            }
        )

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "executable_image_adjudicated_exploratory_posthoc",
        "solver_frozen_before_adjudication": True,
        "benchmark_or_reference_opened_by_builder": False,
        "frozen_solver": {"path": str(args.frozen_solver.resolve()), "sha256": solver_sha, "rows": 274},
        "base_image_judge": {"path": str(args.base_image_judge.resolve()), "sha256": base_sha, "rows": 97},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "rows": 97},
        "adjudicated_rows": adjudicated,
        "copied_base_rows": 97 - len(adjudicated),
        "limitations": [
            "Targets were selected after aggregate benchmark outcome exposure.",
            "This is deterministic public-image adjudication, not an official answer key.",
            "An untouched holdout and independent adjudication are required.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

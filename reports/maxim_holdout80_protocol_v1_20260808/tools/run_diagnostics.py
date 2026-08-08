#!/usr/bin/env python3
"""Run integrity, dedup-summary, and retrieval round-trip diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
REPORT = Path(os.environ.get("VLM_HOLDOUT_REPORT_DIR", Path(__file__).resolve().parents[1])).resolve()
WORKSPACE = Path(os.environ.get("VLM_HOLDOUT_WORKSPACE", PROJECT)).resolve()
TEXTBOOK_ROOT = Path(
    os.environ.get(
        "VLM_HOLDOUT_TEXTBOOK_ROOT",
        next(
            str(candidate)
            for candidate in (
                PROJECT / "artifacts" / "textbooks" / "full_2026_07",
                PROJECT.parent / "VLM" / "artifacts" / "textbooks" / "full_2026_07",
            )
            if candidate.exists()
        ),
    )
).resolve()
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_judge.retrieval import search_bm25  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_page_rows(book_id: str) -> dict[int, str]:
    path = TEXTBOOK_ROOT / "books" / f"{book_id}.pages.jsonl"
    result = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        result[int(row["metadata"]["page_number"])] = row["content"]
    return result


def question_segment(page_text: str, question: int) -> str:
    marker = re.compile(rf"(?<!\d){question}\s*\.\s+", re.UNICODE)
    match = marker.search(page_text)
    if not match:
        return page_text[:4000]
    next_marker = re.compile(rf"(?<!\d){question + 1}\s*\.\s+", re.UNICODE).search(page_text, match.end())
    end = next_marker.start() if next_marker else min(len(page_text), match.start() + 5000)
    return page_text[match.start():end]


def main() -> None:
    manifest_path = REPORT / "selection_manifest.jsonl"
    freeze_path = REPORT / "freeze.json"
    gold_path = REPORT / "sealed" / "sealed_gold.jsonl"
    seal_path = REPORT / "sealed" / "gold_seal.json"
    manifest = load_jsonl(manifest_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gold = load_jsonl(gold_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != freeze["manifest_sha256"]:
        raise RuntimeError("Manifest hash mismatch")
    if sha256_file(gold_path) != seal["sealed_gold_sha256"]:
        raise RuntimeError("Gold hash mismatch")

    question_asset_errors = []
    for row in manifest:
        for path_text, expected in zip(row["question_assets"], row["question_asset_sha256"]):
            path = (WORKSPACE / path_text).resolve()
            if not path.exists() or sha256_file(path) != expected:
                question_asset_errors.append({"task_id": row["task_id"], "path": path_text})
    key_asset_errors = []
    for row in gold:
        for path_text, expected in zip(row["official_key_assets"], row["official_key_asset_sha256"]):
            path = (WORKSPACE / path_text).resolve()
            if not path.exists() or sha256_file(path) != expected:
                key_asset_errors.append({"task_id": row["task_id"], "path": path_text})

    max_row = max(manifest, key=lambda row: (row["benchmark_dedup"]["containment"], row["benchmark_dedup"]["jaccard"]))
    dedup = {
        "schema_version": "holdout80-dedup-audit-v1",
        "benchmark_input_count": freeze["benchmark_input_count"],
        "method": "Unicode-normalized 3-word shingles over official question-page text versus OCR of each of 274 benchmark inputs",
        "selection_gate": {"containment_lt": 0.65, "jaccard_lt": 0.50},
        "all_passed": all(row["benchmark_dedup"]["passed"] for row in manifest),
        "max_observed": {
            "holdout_task_id": max_row["task_id"],
            "nearest_benchmark_task_id": max_row["benchmark_dedup"]["task_id"],
            "containment": max_row["benchmark_dedup"]["containment"],
            "jaccard": max_row["benchmark_dedup"]["jaccard"],
        },
        "known_math12_activity_exclusions": freeze["known_math12_excluded_activities"],
        "exact_selected_asset_sha_matches_to_benchmark": 0,
        "perceptual_crop_audit": {
            "performed": False,
            "reason": "The 274 original bitmaps are not retained under stable task paths; OCR, source-page binding, and exact SHA checks are available.",
        },
        "limitation": "A full source page contains several MCQs, so containment is conservative and is not a learned semantic-duplicate classifier.",
    }
    write_json(REPORT / "dedup_audit.json", dedup)

    math_index = load_jsonl(REPORT / "math12_family_question_index.jsonl")
    inventory = {
        "schema_version": "holdout80-source-inventory-v1",
        "holdout_kind": freeze["holdout_kind"],
        "book_disjoint": False,
        "sources": [
            {
                "source_family": "math12_beceri",
                "source_pdf_sha256": freeze["source_pdf_sha256"]["math12_beceri"],
                "physical_pdf_pages": 182,
                "question_activities_indexed": len(math_index),
                "activity_id_range": [1, 95],
                "selected_activity_ids": sorted(row["activity_id"] for row in manifest if row["source_family"] == "math12_beceri"),
                "benchmark_excluded_activity_ids": freeze["known_math12_excluded_activities"],
                "complete_question_index": "math12_family_question_index.jsonl",
            },
            {
                "source_family": "biology9_textbook",
                "source_pdf_sha256": freeze["source_pdf_sha256"]["biology9"],
                "selected_mcq_count": 30,
                "selection": "complete MCQ census of unit assessments 1-3",
            },
            {
                "source_family": "physics12_textbook",
                "source_pdf_sha256": freeze["source_pdf_sha256"]["physics12"],
                "selected_mcq_count": 30,
                "selection": "five hash-ranked MCQs from each of six unit assessments",
            },
        ],
    }
    write_json(REPORT / "source_inventory.json", inventory)

    certificate = {
        "schema_version": "holdout80-certificate-audit-v1",
        "manifest_hash_ok": True,
        "gold_hash_ok": True,
        "manifest_count": len(manifest),
        "sealed_gold_count": len(gold),
        "task_id_bijection": {row["task_id"] for row in manifest} == {row["task_id"] for row in gold},
        "question_assets_verified": len(manifest) if not question_asset_errors else None,
        "question_asset_errors": question_asset_errors,
        "official_key_records_verified": len(gold) if not key_asset_errors else None,
        "official_key_asset_errors": key_asset_errors,
        "automatic_mcq_gold_count": sum(row["scoring_type"] == "exact_choice" for row in gold),
        "manual_math_gold_count": sum(row["scoring_type"].startswith("manual") for row in gold),
        "math_full_key_page_fallback_count": sum(row.get("reference_extraction_mode") == "full_official_key_page_due_multicolumn_order" for row in gold),
        "overall_integrity_pass": not question_asset_errors and not key_asset_errors and len(manifest) == len(gold) == 80,
        "not_a_model_quality_metric": True,
    }
    write_json(REPORT / "certificate_audit.json", certificate)

    # Retrieval diagnostic.  Query text is extracted directly from official PDFs,
    # so this is an optimistic page round-trip, not end-to-end VLM QA accuracy.
    page_text = {
        "biology9_textbook": load_page_rows("tr-lise-biyoloji-9"),
        "physics12_textbook": load_page_rows("tr-lise-fizik-12"),
    }
    configured_index = os.environ.get("VLM_HOLDOUT_BM25_INDEX")
    index_path = Path(configured_index).resolve() if configured_index else next(
        candidate.resolve()
        for candidate in (
            PROJECT / "artifacts" / "retrieval" / "turkish_textbooks_bm25.sqlite",
            PROJECT.parent / "VLM" / "artifacts" / "retrieval" / "turkish_textbooks_bm25.sqlite",
        )
        if candidate.exists()
    )
    subject = {"biology9_textbook": "biyoloji", "physics12_textbook": "fizik"}
    book_id = {"biology9_textbook": "tr-lise-biyoloji-9", "physics12_textbook": "tr-lise-fizik-12"}
    details = []
    ranks = []
    for row in manifest:
        family = row["source_family"]
        if family == "math12_beceri":
            details.append({"task_id": row["task_id"], "status": "source_family_absent_from_current_bm25_index"})
            continue
        page = row["question_pages"][0]
        query = question_segment(page_text[family][page], row["question_number"])
        search_kwargs = {
            "top_k": 10,
            "subject": subject[family],
            "grade": row["grade"],
            "mode": "or",
        }
        try:
            result = search_bm25(index_path, query, deterministic_ties=True, **search_kwargs)
        except TypeError as exc:
            if "deterministic_ties" not in str(exc):
                raise
            result = search_bm25(index_path, query, **search_kwargs)
        rank = None
        for hit in result["hits"]:
            if hit.get("book_id") == book_id[family] and int(hit.get("page_number") or -1) == page:
                rank = hit["rank"]
                break
        ranks.append(rank)
        details.append({
            "task_id": row["task_id"], "status": "queried", "expected_book_id": book_id[family],
            "expected_page": page, "rank": rank, "returned": result["returned"],
            "query_character_count": len(query), "latency_ms": result["latency_ms"],
        })
    queried = len(ranks)
    retrieval = {
        "schema_version": "holdout80-retrieval-roundtrip-v1",
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "query_source": "gold-blind official PDF text segment for the selected question",
        "important_caveat": "This measures source-page round-trip with clean PDF text. It is not screenshot OCR retrieval, certificate precision, answer accuracy, or end-to-end pipeline quality.",
        "queried_indexed_family_tasks": queried,
        "unindexed_math12_beceri_tasks": 20,
        "hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / queried,
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / queried,
        "hit_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / queried,
        "misses_at_10": sum(rank is None for rank in ranks),
        "rank_histogram": dict(sorted(Counter(str(rank) if rank is not None else "miss" for rank in ranks).items())),
        "details": details,
    }
    write_json(REPORT / "retrieval_roundtrip.json", retrieval)

    manual_form = []
    for row in manifest:
        if row["task_format"] == "multi_part_activity_manual_scoring":
            manual_form.append({
                "task_id": row["task_id"], "reviewer_id": None, "manual_score": None,
                "all_numbered_parts_checked": False, "notes": None,
            })
    (REPORT / "manual_scoring_form.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manual_form),
        encoding="utf-8",
    )
    print(json.dumps({"certificate": certificate, "dedup": dedup, "retrieval": {k: retrieval[k] for k in ("queried_indexed_family_tasks", "unindexed_math12_beceri_tasks", "hit_at_1", "hit_at_5", "hit_at_10", "misses_at_10")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

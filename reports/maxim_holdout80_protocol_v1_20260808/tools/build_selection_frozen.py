#!/usr/bin/env python3
"""Build and freeze a gold-blind 80-task holdout.

This script deliberately has no answer-key constants and never opens answer-key
pages.  It uses only question-page structure, the 274 benchmark inputs (OCR),
and five source bindings known for the Math 12 activity book.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


SCHEMA = "holdout80-selection-v1"
SEED = "holdout80-v1"
REPORT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
BASIC_RAG = WORKSPACE.parent / "VLM_agent_turkish_textbook_basic_rag"
TEXTBOOK_ROOT = WORKSPACE / "artifacts" / "textbooks" / "full_2026_07"
BENCHMARK_OCR = (
    BASIC_RAG
    / "reports"
    / "maxim_document_parser_v1_20260803"
    / "parser_augmented_solver_v1"
    / "parser_artifacts"
    / "parser_results_274.jsonl"
)

MATH_ACTIVITY_START = {
    1: 4, 2: 5, 3: 7, 4: 8, 5: 9, 6: 10, 7: 12, 8: 13, 9: 14,
    10: 15, 11: 16, 12: 18, 13: 19, 14: 20, 15: 22, 16: 23,
    17: 24, 18: 26, 19: 27, 20: 28, 21: 30, 22: 31, 23: 32,
    24: 33, 25: 34, 26: 35, 27: 36, 28: 37, 29: 38, 30: 39,
    31: 40, 32: 42, 33: 44, 34: 45, 35: 46, 36: 47, 37: 49,
    38: 51, 39: 53, 40: 55, 41: 56, 42: 58, 43: 60, 44: 61,
    45: 62, 46: 63, 47: 64, 48: 65, 49: 67, 50: 68, 51: 70,
    52: 71, 53: 73, 54: 74, 55: 75, 56: 77, 57: 79, 58: 80,
    59: 82, 60: 83, 61: 85, 62: 86, 63: 88, 64: 89, 65: 91,
    66: 93, 67: 95, 68: 96, 69: 97, 70: 98, 71: 100, 72: 101,
    73: 102, 74: 103, 75: 104, 76: 105, 77: 106, 78: 107,
    79: 108, 80: 110, 81: 111, 82: 112, 83: 113, 84: 115,
    85: 116, 86: 117, 87: 118, 88: 119, 89: 120, 90: 121,
    91: 123, 92: 125, 93: 127, 94: 128, 95: 129,
}

# Source bindings are input-only exclusions.  No benchmark answer or outcome is
# read.  Page numbers are physical 1-based PDF pages.
KNOWN_MATH12_BENCHMARK = [
    {"task_id": "val_0054", "activity_id": 3, "question_pages": [7]},
    {"task_id": "val_0055", "activity_id": 17, "question_pages": [24]},
    {"task_id": "val_0056", "activity_id": 88, "question_pages": [119]},
    {"task_id": "val_0057", "activity_id": 43, "question_pages": [60]},
    {"task_id": "val_0058", "activity_id": 31, "question_pages": [40, 41]},
]
MATH_EXCLUDED = {row["activity_id"] for row in KNOWN_MATH12_BENCHMARK}

MATH_CATEGORY_QUOTAS = {
    "exponential_logarithmic": (range(1, 17), 3),
    "sequences": (range(17, 30), 3),
    "trigonometry": (range(30, 37), 3),
    "transformations": (range(37, 43), 2),
    "derivative": (range(43, 72), 3),
    "integral": (range(72, 90), 3),
    "analytic_geometry": (range(90, 96), 3),
}

BIO_PAGE_MAP = {
    1: {**{q: 63 for q in range(1, 5)}, **{q: 64 for q in range(5, 9)}, **{q: 65 for q in range(9, 13)}},
    2: {**{q: 108 for q in range(1, 4)}, **{q: 109 for q in range(4, 8)}, **{q: 110 for q in range(8, 13)}},
    3: {q: 154 for q in range(1, 7)},
}

PHYSICS_POOLS = {
    1: range(34, 59),
    2: range(24, 39),
    3: range(28, 39),
    4: range(30, 59),
    5: range(21, 41),
    6: range(30, 47),
}
PHYSICS_PAGE_MAP = {
    1: {**{q: 74 for q in range(34, 38)}, **{q: 75 for q in range(38, 42)}, **{q: 76 for q in range(42, 50)}, **{q: 77 for q in range(50, 55)}, **{q: 78 for q in range(55, 59)}},
    2: {**{q: 100 for q in range(24, 28)}, **{q: 101 for q in range(28, 35)}, **{q: 102 for q in range(35, 39)}},
    3: {**{q: 138 for q in range(28, 31)}, **{q: 139 for q in range(31, 38)}, 38: 140},
    4: {**{q: 180 for q in range(30, 33)}, **{q: 181 for q in range(33, 37)}, **{q: 182 for q in range(37, 47)}, **{q: 183 for q in range(47, 54)}, **{q: 184 for q in range(54, 59)}},
    5: {**{q: 224 for q in range(21, 27)}, **{q: 225 for q in range(27, 34)}, **{q: 226 for q in range(34, 41)}},
    6: {**{q: 253 for q in range(30, 35)}, **{q: 254 for q in range(35, 47)}},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path("..") .joinpath(path.resolve().relative_to(WORKSPACE.parent.resolve())).as_posix()


def stable_rank(namespace: str) -> str:
    return sha256_bytes(f"{SEED}|{namespace}".encode("utf-8"))


def load_page_rows(book_id: str) -> dict[int, dict]:
    path = TEXTBOOK_ROOT / "books" / f"{book_id}.pages.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[int(row["metadata"]["page_number"])] = row
    return rows


def benchmark_texts() -> tuple[list[dict], set[str]]:
    rows = []
    hashes: set[str] = set()
    with BENCHMARK_OCR.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            parts = []
            for image in raw.get("images", []):
                if image.get("image_sha256"):
                    hashes.add(image["image_sha256"])
                for block in image.get("parsing_res_list", []):
                    content = block.get("block_content")
                    if content:
                        parts.append(content)
            rows.append({"task_id": raw["task_id"], "text": "\n".join(parts)})
    if len(rows) != 274:
        raise RuntimeError(f"Expected 274 benchmark inputs, got {len(rows)}")
    return rows, hashes


def word_shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def max_near_duplicate(text: str, benchmark: list[dict]) -> dict:
    a = word_shingles(text)
    best = {"task_id": None, "containment": 0.0, "jaccard": 0.0}
    if not a:
        return best
    for row in benchmark:
        b = word_shingles(row["text"])
        if not b:
            continue
        overlap = len(a & b)
        containment = overlap / min(len(a), len(b))
        jaccard = overlap / len(a | b)
        if (containment, jaccard) > (best["containment"], best["jaccard"]):
            best = {"task_id": row["task_id"], "containment": round(containment, 6), "jaccard": round(jaccard, 6)}
    return best


def passes_dedup(result: dict) -> bool:
    # Conservative crop-to-page containment gate, plus a symmetric similarity gate.
    return result["containment"] < 0.65 and result["jaccard"] < 0.50


def find_math_pdf() -> Path:
    matches = list((BASIC_RAG / "tmp" / "remaining_official_source_audit" / "pdfs").glob("matematik 12*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Math12 Beceri PDF, found {len(matches)}")
    return matches[0]


def render_pdf_page(pdf: Path, page: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = destination.with_suffix("")
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm is not available")
    executable_path = Path(executable)
    if executable_path.suffix.casefold() in {".cmd", ".bat"}:
        native = executable_path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
        if native.exists():
            executable = str(native)
    command = [
        executable, "-f", str(page), "-l", str(page), "-r", "144",
        "-jpeg", "-jpegopt", "quality=88,progressive=y,optimize=y",
        "-singlefile", str(pdf), str(prefix),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not destination.exists():
        raise RuntimeError(f"Renderer did not create {destination}")


def convert_png_to_jpeg(source: Path, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, "JPEG", quality=88, optimize=True, progressive=True)


def build() -> None:
    manifest_path = REPORT / "selection_manifest.jsonl"
    freeze_path = REPORT / "freeze.json"
    if manifest_path.exists() or freeze_path.exists():
        raise RuntimeError("Selection is already frozen; remove the report directory only if intentionally rebuilding v1")

    benchmark, benchmark_hashes = benchmark_texts()
    math_pdf = find_math_pdf()
    sources = {
        "math12_beceri": math_pdf,
        "biology9": TEXTBOOK_ROOT / "original_pdfs" / "tr-lise-biyoloji-9.pdf",
        "physics12": TEXTBOOK_ROOT / "original_pdfs" / "tr-lise-fizik-12.pdf",
    }
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    math_reader = PdfReader(str(math_pdf))
    if len(math_reader.pages) != 182:
        raise RuntimeError(f"Unexpected Math12 page count: {len(math_reader.pages)}")
    math_text = {p: (math_reader.pages[p - 1].extract_text() or "") for p in range(1, 131)}
    page_rows = {
        "biology9": load_page_rows("tr-lise-biyoloji-9"),
        "physics12": load_page_rows("tr-lise-fizik-12"),
    }

    # The complete 95-activity question index is built before any answer key is opened.
    math_index = []
    for activity_id in range(1, 96):
        start = MATH_ACTIVITY_START[activity_id]
        end = MATH_ACTIVITY_START.get(activity_id + 1, 131) - 1
        pages = list(range(start, end + 1))
        math_index.append({
            "activity_id": activity_id,
            "question_pages": pages,
            "question_text_sha256": sha256_bytes("\n".join(math_text[p] for p in pages).encode("utf-8")),
            "benchmark_excluded": activity_id in MATH_EXCLUDED,
            "benchmark_task_ids": [r["task_id"] for r in KNOWN_MATH12_BENCHMARK if r["activity_id"] == activity_id],
        })
    write_jsonl(REPORT / "math12_family_question_index.jsonl", math_index)
    write_jsonl(REPORT / "known_math12_benchmark_bindings.jsonl", KNOWN_MATH12_BENCHMARK)

    selected_math: list[tuple[str, int, dict]] = []
    for category, (activity_range, quota) in MATH_CATEGORY_QUOTAS.items():
        ranked = sorted(
            (stable_rank(f"math12-beceri|activity:{activity_id}"), activity_id)
            for activity_id in activity_range
            if activity_id not in MATH_EXCLUDED
        )
        accepted = []
        for rank, activity_id in ranked:
            index = math_index[activity_id - 1]
            text = "\n".join(math_text[p] for p in index["question_pages"])
            dedup = max_near_duplicate(text, benchmark)
            if passes_dedup(dedup):
                accepted.append((category, activity_id, {"rank": rank, "dedup": dedup}))
            if len(accepted) == quota:
                break
        if len(accepted) != quota:
            raise RuntimeError(f"Could not fill Math12 category {category}: {len(accepted)}/{quota}")
        selected_math.extend(accepted)

    manifest: list[dict] = []
    # Render each page once.  Full activity pages are retained; no cherry-picked crop.
    for category, activity_id, selection in selected_math:
        index = math_index[activity_id - 1]
        assets = []
        page_hashes = []
        for page in index["question_pages"]:
            asset = REPORT / "assets" / "questions" / "math12_beceri" / f"page-{page:04d}.jpg"
            if not asset.exists():
                render_pdf_page(math_pdf, page, asset)
            digest = sha256_file(asset)
            if digest in benchmark_hashes:
                raise RuntimeError(f"Exact benchmark image duplicate: Math12 page {page}")
            assets.append(rel(asset))
            page_hashes.append(digest)
        manifest.append({
            "schema_version": SCHEMA,
            "task_id": f"h80-math12-a{activity_id:03d}",
            "subject": "mathematics",
            "grade": 12,
            "language": "tr",
            "task_format": "multi_part_activity_manual_scoring",
            "source_family": "math12_beceri",
            "source_title": "Matematik 12 Beceri Temelli Etkinlik Kitabı",
            "source_pdf": rel(math_pdf),
            "source_pdf_sha256": source_hashes["math12_beceri"],
            "activity_id": activity_id,
            "structural_stratum": category,
            "question_pages": index["question_pages"],
            "question_assets": assets,
            "question_asset_sha256": page_hashes,
            "question_text_sha256": index["question_text_sha256"],
            "prompt": "Görsellerdeki etkinliğin bütün numaralı sorularını çözünüz. Sonuçları soru numaralarıyla yazınız.",
            "selection_rank_sha256": selection["rank"],
            "benchmark_dedup": {**selection["dedup"], "passed": True, "containment_threshold": 0.65, "jaccard_threshold": 0.50},
        })

    # Biology: all 30 MCQs from all three unit assessments, a structural census.
    for unit, mapping in BIO_PAGE_MAP.items():
        for question, page in sorted(mapping.items()):
            text = page_rows["biology9"][page]["content"]
            dedup = max_near_duplicate(text, benchmark)
            if not passes_dedup(dedup):
                raise RuntimeError(f"Biology page {page} is too close to benchmark {dedup}")
            source_asset = TEXTBOOK_ROOT / "page_images" / "tr-lise-biyoloji-9" / f"page-{page:04d}.png"
            asset = REPORT / "assets" / "questions" / "biology9" / f"page-{page:04d}.jpg"
            if not asset.exists():
                convert_png_to_jpeg(source_asset, asset)
            digest = sha256_file(asset)
            if digest in benchmark_hashes:
                raise RuntimeError(f"Exact benchmark image duplicate: Biology page {page}")
            manifest.append({
                "schema_version": SCHEMA,
                "task_id": f"h80-bio9-u{unit}-q{question:02d}",
                "subject": "biology",
                "grade": 9,
                "language": "tr",
                "task_format": "multiple_choice_ABCDE",
                "source_family": "biology9_textbook",
                "source_title": "9. Sınıf Biyoloji Ders Kitabı",
                "source_pdf": rel(sources["biology9"]),
                "source_pdf_sha256": source_hashes["biology9"],
                "unit": unit,
                "question_number": question,
                "question_pages": [page],
                "question_assets": [rel(asset)],
                "question_asset_sha256": [digest],
                "question_text_sha256": sha256_bytes(text.encode("utf-8")),
                "prompt": f"Sayfadaki {question}. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C, D veya E yazınız.",
                "selection_rank_sha256": stable_rank(f"biology9|unit:{unit}|question:{question}"),
                "benchmark_dedup": {**dedup, "passed": True, "containment_threshold": 0.65, "jaccard_threshold": 0.50},
            })

    # Physics: five gold-blind, hash-ranked questions per unit.
    for unit, pool in PHYSICS_POOLS.items():
        accepted = []
        for rank, question in sorted(
            (stable_rank(f"physics12|unit:{unit}|question:{q}"), q) for q in pool
        ):
            page = PHYSICS_PAGE_MAP[unit][question]
            text = page_rows["physics12"][page]["content"]
            dedup = max_near_duplicate(text, benchmark)
            if passes_dedup(dedup):
                accepted.append((rank, question, page, text, dedup))
            if len(accepted) == 5:
                break
        if len(accepted) != 5:
            raise RuntimeError(f"Could not fill Physics unit {unit}: {len(accepted)}/5")
        for rank, question, page, text, dedup in accepted:
            source_asset = TEXTBOOK_ROOT / "page_images" / "tr-lise-fizik-12" / f"page-{page:04d}.png"
            asset = REPORT / "assets" / "questions" / "physics12" / f"page-{page:04d}.jpg"
            if not asset.exists():
                convert_png_to_jpeg(source_asset, asset)
            digest = sha256_file(asset)
            if digest in benchmark_hashes:
                raise RuntimeError(f"Exact benchmark image duplicate: Physics page {page}")
            manifest.append({
                "schema_version": SCHEMA,
                "task_id": f"h80-phys12-u{unit}-q{question:02d}",
                "subject": "physics",
                "grade": 12,
                "language": "tr",
                "task_format": "multiple_choice_ABCDE",
                "source_family": "physics12_textbook",
                "source_title": "12. Sınıf Fizik Ders Kitabı",
                "source_pdf": rel(sources["physics12"]),
                "source_pdf_sha256": source_hashes["physics12"],
                "unit": unit,
                "question_number": question,
                "question_pages": [page],
                "question_assets": [rel(asset)],
                "question_asset_sha256": [digest],
                "question_text_sha256": sha256_bytes(text.encode("utf-8")),
                "prompt": f"Sayfadaki {question}. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C, D veya E yazınız.",
                "selection_rank_sha256": rank,
                "benchmark_dedup": {**dedup, "passed": True, "containment_threshold": 0.65, "jaccard_threshold": 0.50},
            })

    manifest.sort(key=lambda row: row["task_id"])
    counts = {
        "total": len(manifest),
        "math12_beceri": sum(r["source_family"] == "math12_beceri" for r in manifest),
        "biology9": sum(r["source_family"] == "biology9_textbook" for r in manifest),
        "physics12": sum(r["source_family"] == "physics12_textbook" for r in manifest),
    }
    if counts != {"total": 80, "math12_beceri": 20, "biology9": 30, "physics12": 30}:
        raise RuntimeError(f"Unexpected split: {counts}")
    if len({r["task_id"] for r in manifest}) != 80:
        raise RuntimeError("Duplicate holdout task_id")

    write_jsonl(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    freeze = {
        "schema_version": "holdout80-freeze-v1",
        "status": "FROZEN_BEFORE_GOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_kind": "same_book_new_question_holdout",
        "book_disjoint": False,
        "selection_seed": SEED,
        "selection_policy": "source_structure_plus_hash_rank_only",
        "benchmark_outcomes_accessed": False,
        "benchmark_gold_accessed": False,
        "benchmark_input_count": 274,
        "benchmark_input_artifact": rel(BENCHMARK_OCR),
        "benchmark_input_artifact_sha256": sha256_file(BENCHMARK_OCR),
        "known_math12_excluded_activities": sorted(MATH_EXCLUDED),
        "manifest": rel(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_count": len(manifest),
        "counts": counts,
        "source_pdf_sha256": source_hashes,
        "selection_script_sha256": sha256_file(Path(__file__)),
        "notes": [
            "The split measures transfer to unseen questions inside already available books.",
            "It must not be described as book-disjoint or domain-disjoint.",
            "Gold sealing is a separate command that verifies this manifest hash first.",
        ],
    }
    write_json(freeze_path, freeze)
    print(json.dumps({"manifest": str(manifest_path), "freeze": str(freeze_path), "manifest_sha256": manifest_sha, "counts": counts}, ensure_ascii=False, indent=2))


def verify() -> None:
    manifest_path = REPORT / "selection_manifest.jsonl"
    freeze = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    actual = sha256_file(manifest_path)
    if actual != freeze["manifest_sha256"]:
        raise RuntimeError(f"Manifest hash mismatch: expected {freeze['manifest_sha256']}, got {actual}")
    rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    if len(rows) != freeze["manifest_count"]:
        raise RuntimeError("Manifest count mismatch")
    for row in rows:
        if "answer" in row or "gold" in row or "reference_solution" in row:
            raise RuntimeError(f"Gold-like field leaked into manifest: {row['task_id']}")
        for path_text, expected in zip(row["question_assets"], row["question_asset_sha256"]):
            path = (WORKSPACE / path_text).resolve()
            if sha256_file(path) != expected:
                raise RuntimeError(f"Question asset hash mismatch: {path}")
    print(json.dumps({"ok": True, "manifest_sha256": actual, "count": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify"])
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

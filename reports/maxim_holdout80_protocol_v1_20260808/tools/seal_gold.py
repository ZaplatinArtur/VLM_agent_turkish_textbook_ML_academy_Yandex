#!/usr/bin/env python3
"""Seal official answer keys after selection_manifest.jsonl has been frozen."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


REPORT = Path(os.environ.get("VLM_HOLDOUT_REPORT_DIR", Path(__file__).resolve().parents[1])).resolve()
WORKSPACE = Path(os.environ.get("VLM_HOLDOUT_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()

BIO_KEYS = {
    1: dict(enumerate("DEDCBBBD BADA".replace(" ", ""), start=1)),
    2: dict(enumerate("EDDDD A CDEBDE".replace(" ", ""), start=1)),
    3: dict(enumerate("ADB AAA".replace(" ", ""), start=1)),
}
BIO_KEY_PAGE = {1: 158, 2: 160, 3: 162}

PHYSICS_KEYS = {
    1: dict(zip(range(34, 59), "EACBBADBDAC EEDBCE CDEE BDA B".replace(" ", ""))),
    2: dict(zip(range(24, 43), "ADDEADACBE EBDBC DACE".replace(" ", ""))),
    3: dict(zip(range(28, 39), "DDBDCECBCAA")),
    4: dict(zip(range(30, 59), "ECDDE EABCC BBDCEABDBE CDACABD AA".replace(" ", ""))),
    5: dict(zip(range(21, 41), "EEDAEDCABC BCCADEBBDC".replace(" ", ""))),
    6: dict(zip(range(30, 47), "DDCEECDEBE DDEDEDA".replace(" ", ""))),
}
PHYSICS_KEY_PAGE = {1: 263, 2: 264, 3: 264, 4: 264, 5: 265, 6: 265}

# Official solution pages for all selected activity IDs.  These locators were
# not used by build_selection.py.
MATH_KEY_PAGE = {
    12: [136], 14: [137], 15: [138], 18: [139], 23: [142], 26: [143],
    30: [145], 32: [146], 36: [149], 37: [149], 39: [151], 53: [156],
    57: [159], 62: [161], 73: [168], 77: [169], 87: [173],
    90: [175, 176], 91: [176], 93: [177],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path("..").joinpath(path.resolve().relative_to(WORKSPACE.parent.resolve())).as_posix()


def render_pdf_page(pdf: Path, page: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm is not available")
    executable_path = Path(executable)
    if executable_path.suffix.casefold() in {".cmd", ".bat"}:
        native = executable_path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
        if native.exists():
            executable = str(native)
    subprocess.run(
        [executable, "-f", str(page), "-l", str(page), "-r", "144", "-jpeg",
         "-jpegopt", "quality=88,progressive=y,optimize=y", "-singlefile",
         str(pdf), str(destination.with_suffix(""))],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def extract_activity_solution(page_text: str, activity_id: int) -> str:
    """Best-effort isolate one activity; retain full page when PDF order is ambiguous."""
    marker = re.compile(rf"Etkinlik\s+No\.?\s*:\s*{activity_id}(?!\d)", re.IGNORECASE)
    match = marker.search(page_text)
    if not match:
        return page_text.strip()
    next_match = re.search(r"Etkinlik\s+No\.?\s*:\s*\d+", page_text[match.end():], re.IGNORECASE)
    end = match.end() + next_match.start() if next_match else len(page_text)
    return page_text[match.start():end].strip()


def main() -> None:
    freeze_path = REPORT / "freeze.json"
    manifest_path = REPORT / "selection_manifest.jsonl"
    if not freeze_path.exists() or not manifest_path.exists():
        raise RuntimeError("Run build_selection.py build first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != freeze["manifest_sha256"]:
        raise RuntimeError("Refusing to open gold: frozen manifest hash does not match")
    if freeze.get("status") != "FROZEN_BEFORE_GOLD":
        raise RuntimeError("Unexpected freeze status")
    output = REPORT / "sealed" / "sealed_gold.jsonl"
    if output.exists():
        old_seal_path = REPORT / "sealed" / "gold_seal.json"
        if not old_seal_path.exists():
            raise RuntimeError("Existing sealed_gold.jsonl has no seal; refusing to overwrite")
        old_seal = json.loads(old_seal_path.read_text(encoding="utf-8"))
        if old_seal.get("frozen_manifest_sha256") != actual_manifest_sha:
            raise RuntimeError("Existing gold belongs to another manifest; refusing to overwrite")

    rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    pdfs = {family: (WORKSPACE / next(r["source_pdf"] for r in rows if r["source_family"] == family)).resolve()
            for family in {r["source_family"] for r in rows}}
    readers = {family: PdfReader(str(path)) for family, path in pdfs.items()}
    sealed = []
    for row in rows:
        family = row["source_family"]
        if family == "biology9_textbook":
            unit, question = row["unit"], row["question_number"]
            answer = BIO_KEYS[unit][question]
            key_pages = [BIO_KEY_PAGE[unit]]
            scoring = "exact_choice"
            reference = None
            extraction_mode = "official_mcq_table_transcription"
        elif family == "physics12_textbook":
            unit, question = row["unit"], row["question_number"]
            answer = PHYSICS_KEYS[unit][question]
            key_pages = [PHYSICS_KEY_PAGE[unit]]
            scoring = "exact_choice"
            reference = None
            extraction_mode = "official_mcq_table_transcription"
        elif family == "math12_beceri":
            activity = row["activity_id"]
            answer = None
            key_pages = MATH_KEY_PAGE[activity]
            page_texts = [readers[family].pages[p - 1].extract_text() or "" for p in key_pages]
            reference = "\n\n".join(extract_activity_solution(text, activity) for text in page_texts)
            extraction_mode = "activity_marker_segment"
            if len(reference.strip()) < 100:
                # Some official key pages place several activity markers before
                # multi-column text.  In that case a short marker-only slice is
                # unsafe, so retain the complete official page for a human scorer.
                reference = "\n\n".join(page_texts).strip()
                extraction_mode = "full_official_key_page_due_multicolumn_order"
            scoring = "manual_against_official_solution"
        else:
            raise RuntimeError(f"Unknown family {family}")

        key_assets = []
        key_asset_hashes = []
        for page in key_pages:
            asset = REPORT / "sealed" / "key_pages" / family / f"page-{page:04d}.jpg"
            if not asset.exists():
                render_pdf_page(pdfs[family], page, asset)
            key_assets.append(rel(asset))
            key_asset_hashes.append(sha256_file(asset))
        payload = {
            "schema_version": "holdout80-sealed-gold-v1",
            "task_id": row["task_id"],
            "scoring_type": scoring,
            "official_answer": answer,
            "official_reference_solution": reference,
            "reference_extraction_mode": extraction_mode,
            "official_key_pages": key_pages,
            "official_key_assets": key_assets,
            "official_key_asset_sha256": key_asset_hashes,
            "source_pdf_sha256": row["source_pdf_sha256"],
            "frozen_manifest_sha256": actual_manifest_sha,
        }
        payload["gold_payload_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        sealed.append(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(canonical_json(row) + "\n" for row in sealed), encoding="utf-8")
    seal = {
        "schema_version": "holdout80-gold-seal-v1",
        "frozen_manifest_sha256": actual_manifest_sha,
        "sealed_gold": rel(output),
        "sealed_gold_sha256": sha256_file(output),
        "count": len(sealed),
        "automatic_exact_choice_count": sum(r["scoring_type"] == "exact_choice" for r in sealed),
        "manual_math_activity_count": sum(r["scoring_type"].startswith("manual") for r in sealed),
        "warning": "Math activity solutions require blinded manual scoring; do not report strict overall accuracy from MCQ only.",
    }
    seal_path = REPORT / "sealed" / "gold_seal.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(seal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

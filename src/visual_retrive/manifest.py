"""Build page-bundle manifests from scraped OdevJet books."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from retrieve.metadata import infer_textbook_metadata

from .paths import BOOKS_DIR, CATALOG_DIR, VISUAL_RETRIVE_DIR, ensure_visual_retrive_dirs

_BOILERPLATE_CUT = re.compile(
    r"(Henüz onaylanmış öğrenci çözümü|Bu sayfadaki ders kitabı cevapları)",
    flags=re.IGNORECASE,
)


def page_id_for(book_slug: str, page_number: int) -> str:
    return f"{book_slug}:{page_number:04d}"


def clean_answer_text(text: str, *, max_chars: int = 4_000) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    match = _BOILERPLATE_CUT.search(text)
    if match:
        text = text[: match.start()].strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.relative_to(VISUAL_RETRIVE_DIR).as_posix()


def iter_answer_json_files(books_dir: Path | None = None) -> Iterator[Path]:
    root = books_dir or BOOKS_DIR
    if not root.is_dir():
        return
    yield from sorted(root.glob("*/answers/*.json"))


def build_page_bundle(answer_json_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(answer_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    book_slug = str(payload.get("book_slug") or answer_json_path.parents[1].name)
    try:
        page_number = int(payload.get("page_number") or answer_json_path.stem)
    except (TypeError, ValueError):
        return None

    inferred = infer_textbook_metadata(book_slug)
    page_image = BOOKS_DIR / book_slug / "pages" / f"{page_number:04d}.jpg"
    answer_text = clean_answer_text(str(payload.get("answer_text") or ""))
    if not answer_text:
        txt_path = answer_json_path.with_suffix(".txt")
        if txt_path.is_file():
            answer_text = clean_answer_text(txt_path.read_text(encoding="utf-8"))

    answer_image_paths = [
        str(path)
        for path in (payload.get("answer_image_paths") or [])
        if isinstance(path, str)
    ]
    # Keep only files that still exist.
    existing_answer_images = []
    for rel in answer_image_paths:
        absolute = VISUAL_RETRIVE_DIR / rel
        if absolute.is_file():
            existing_answer_images.append(rel)

    page_image_rel = _rel_if_exists(page_image)
    if page_image_rel is None and payload.get("page_image_path"):
        candidate = VISUAL_RETRIVE_DIR / str(payload["page_image_path"])
        page_image_rel = _rel_if_exists(candidate)

    has_solution = bool(payload.get("has_solution")) and bool(
        answer_text or existing_answer_images
    )
    kinds: list[str] = []
    if answer_text:
        kinds.append("text")
    if existing_answer_images:
        kinds.append("image")

    return {
        "page_id": page_id_for(book_slug, page_number),
        "book_slug": book_slug,
        "page_number": page_number,
        "grade": inferred.get("grade"),
        "subject": inferred.get("subject"),
        "page_url": payload.get("page_url"),
        "page_image": page_image_rel,
        "page_image_sha256": _file_sha256(page_image) if page_image_rel else None,
        "answer_text": answer_text,
        "answer_text_path": _rel_if_exists(answer_json_path.with_suffix(".txt")),
        "answer_meta_path": answer_json_path.relative_to(VISUAL_RETRIVE_DIR).as_posix(),
        "answer_image_paths": existing_answer_images,
        "answer_kinds": kinds,
        "has_solution": has_solution,
        "no_solution": not has_solution,
        "retrieval_text": answer_text,
    }


def build_manifest(
    *,
    output_path: Path | None = None,
    require_page_image: bool = False,
    require_solution: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_visual_retrive_dirs()
    out = output_path or (CATALOG_DIR / "page_bundles.jsonl")
    rows: list[dict[str, Any]] = []
    for path in iter_answer_json_files():
        bundle = build_page_bundle(path)
        if bundle is None:
            continue
        if require_page_image and not bundle.get("page_image"):
            continue
        if require_solution and not bundle.get("has_solution"):
            continue
        rows.append(bundle)
        if limit is not None and len(rows) >= limit:
            break

    rows.sort(key=lambda row: (row["book_slug"], row["page_number"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(out),
        "pages": len(rows),
        "with_page_image": sum(1 for row in rows if row.get("page_image")),
        "with_solution": sum(1 for row in rows if row.get("has_solution")),
        "with_answer_text": sum(1 for row in rows if row.get("answer_text")),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def read_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    manifest_path = path or (CATALOG_DIR / "page_bundles.jsonl")
    if not manifest_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

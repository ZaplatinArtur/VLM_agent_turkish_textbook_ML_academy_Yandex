from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schemas.media import ImageRef
from schemas.retrieve import RetrievedChunk


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mime_type(url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.casefold()
    if suffix == ".webp":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(url)
    return guessed or "application/octet-stream"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("books"), list):
        raise ValueError("manifest must be an object containing a books list")
    books: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for index, raw in enumerate(payload["books"]):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest books[{index}] must be an object")
        book_id = str(raw.get("book_id") or "").strip()
        slug = str(raw.get("slug") or "").strip()
        if not book_id or not slug:
            raise ValueError(
                f"manifest books[{index}] must contain book_id and slug"
            )
        if book_id in seen_ids:
            raise ValueError(f"duplicate manifest book_id: {book_id}")
        if slug in seen_slugs:
            raise ValueError(f"duplicate manifest slug: {slug}")
        seen_ids.add(book_id)
        seen_slugs.add(slug)
        books.append({**raw, "book_id": book_id, "slug": slug})
    return books


def _read_selected_pages(
    pages_path: Path,
    books: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    selected_ids = {book["book_id"] for book in books}
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with pages_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{pages_path}:{line_number}: {exc}") from exc
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                continue
            book_id = str(metadata.get("kitap_id") or "").strip()
            if book_id in selected_ids:
                selected[book_id].append(row)
    return selected


def _page_number(row: dict[str, Any]) -> int:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("page record lacks metadata")
    value = metadata.get("sayfa_no")
    try:
        page = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid page number: {value!r}") from exc
    if page < 0:
        raise ValueError(f"page number must be non-negative: {page}")
    return page


def _image_urls(
    row: dict[str, Any],
    *,
    book: dict[str, Any],
    page: int,
) -> list[str]:
    template = str(book.get("image_url_template") or "").strip()
    if template:
        return [template.format(page=page, page04=f"{page:04d}")]
    metadata = row.get("metadata") or {}
    raw_urls = metadata.get("image_urls")
    return [str(url) for url in raw_urls or [] if str(url).strip()]


def _as_chunk(
    row: dict[str, Any],
    *,
    book: dict[str, Any],
) -> RetrievedChunk:
    source_metadata = row.get("metadata") or {}
    provenance = row.get("provenance") or {}
    page = _page_number(row)
    slug = str(book["slug"])
    chunk_id = f"{slug}:{page:04d}"
    images = [
        ImageRef(
            image_id=chunk_id if index == 0 else f"{chunk_id}:img{index + 1}",
            format="url",
            data=url,
            mime_type=_mime_type(url),
        )
        for index, url in enumerate(
            _image_urls(row, book=book, page=page)
        )
    ]
    metadata = {
        "textbook": slug,
        "page": page,
        "grade": book.get("grade", source_metadata.get("sinif")),
        "subject": book.get("subject", source_metadata.get("ders")),
        "publisher": book.get("publisher", source_metadata.get("yayinevi")),
        "author": source_metadata.get("author"),
        "source": "odevjet",
        "source_url": source_metadata.get("url"),
        "source_book_id": str(book["book_id"]),
        "source_book_title": source_metadata.get("kitap_title"),
        "source_page_id": row.get("id"),
        "source_page_hash": provenance.get("page_hash"),
        "source_content_hash": provenance.get("content_hash"),
        "source_variants": provenance.get("source_variants"),
        "source_conflicting_id": bool(provenance.get("conflicting_id")),
        "source_low_information": bool(provenance.get("low_information")),
        "source_boilerplate": bool(provenance.get("boilerplate")),
        "corpus_snapshot": book.get("corpus_snapshot"),
        "ingest_method": "canonical_odevjet_page_export_v1",
    }
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=str(row.get("content") or "").strip(),
        images=images,
        score=0.0,
        metadata=metadata,
    )


def export_books(
    *,
    pages_path: Path,
    manifest_path: Path,
    output_dir: Path,
    report_path: Path | None = None,
    overwrite: bool = False,
    allow_page_count_mismatch: bool = False,
) -> dict[str, Any]:
    books = _read_manifest(manifest_path)
    selected = _read_selected_pages(pages_path, books)
    prepared: list[tuple[dict[str, Any], list[RetrievedChunk]]] = []
    validation: list[dict[str, Any]] = []

    for book in books:
        rows = selected.get(book["book_id"], [])
        if not rows:
            raise ValueError(f"no pages found for book_id={book['book_id']}")
        chunks = [_as_chunk(row, book=book) for row in rows]
        chunks.sort(key=lambda chunk: int(chunk.metadata["page"]))
        page_numbers = [int(chunk.metadata["page"]) for chunk in chunks]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError(f"duplicate page number for slug={book['slug']}")
        expected = book.get("expected_pages")
        mismatch = expected is not None and len(chunks) != int(expected)
        if mismatch and not allow_page_count_mismatch:
            raise ValueError(
                f"page count mismatch for {book['slug']}: "
                f"expected={expected}, actual={len(chunks)}"
            )
        validation.append(
            {
                "book_id": book["book_id"],
                "slug": book["slug"],
                "grade": book.get("grade"),
                "subject": book.get("subject"),
                "expected_pages": expected,
                "exported_pages": len(chunks),
                "page_count_mismatch": mismatch,
                "min_page": min(page_numbers),
                "max_page": max(page_numbers),
                "nonempty_text_pages": sum(bool(chunk.text) for chunk in chunks),
                "image_pages": sum(bool(chunk.images) for chunk in chunks),
                "low_information_pages": sum(
                    bool(chunk.metadata["source_low_information"])
                    for chunk in chunks
                ),
            }
        )
        prepared.append((book, chunks))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for book, chunks in prepared:
        output_path = output_dir / f"{book['slug']}.jsonl"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output_path}"
            )
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        temporary_path.replace(output_path)
        outputs.append(
            {
                "slug": book["slug"],
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
            }
        )

    report = {
        "schema_version": 1,
        "ingest_method": "canonical_odevjet_page_export_v1",
        "input_pages": str(pages_path),
        "input_pages_sha256": _sha256(pages_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "books": len(books),
        "pages": sum(row["exported_pages"] for row in validation),
        "validation": validation,
        "outputs": outputs,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected OdevJet canonical pages into the per-book "
            "RetrievedChunk JSONL format consumed by educational chunking."
        )
    )
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-page-count-mismatch", action="store_true")
    args = parser.parse_args()
    report = export_books(
        pages_path=args.pages,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        report_path=args.report,
        overwrite=args.overwrite,
        allow_page_count_mismatch=args.allow_page_count_mismatch,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

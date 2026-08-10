"""Safely stage official textbook PDFs declared by a manifest.

This downloader deliberately stops at an immutable staging directory.  It
does not call the PDF import adapter and never writes into ``data/``.
Existing PDFs, provenance files, and canonical ``.part`` files are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse
from uuid import uuid4

import requests


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pdf$", re.IGNORECASE)
_PDF_MAGIC = b"%PDF-"
_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class TextbookSource:
    source_id: str
    slug: str
    grade: int | None
    subject: str
    source: str
    pdf_url: str
    filename: str
    expected_sha256: str | None
    manifest_entry: Mapping[str, Any]


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def _nested_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_filename(value: str) -> str:
    filename = value.strip()
    if (
        not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
        or _SAFE_FILENAME_RE.fullmatch(filename) is None
    ):
        raise ValueError(f"unsafe PDF filename: {value!r}")
    return filename


def _validate_url(value: str, *, source_id: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"book {source_id!r} has no absolute HTTP(S) PDF URL"
        )
    return value


def _normalize_expected_sha256(value: Any, *, source_id: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"book {source_id!r} has invalid expected_sha256"
        )
    return normalized


def _manifest_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, Mapping):
        raw_entries = None
        for key in ("entries", "books", "textbooks", "documents", "sources"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                raw_entries = candidate
                break
        if raw_entries is None:
            raise ValueError(
                "manifest must contain an entries or books list "
                "(textbooks/documents/sources are accepted aliases)"
            )
    else:
        raise ValueError("manifest must be a JSON object or list")

    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest entry {index} must be an object")
        entries.append(entry)
    return entries


def _parse_book(entry: Mapping[str, Any], *, index: int) -> TextbookSource:
    source_meta = _nested_mapping(entry.get("source"))
    download_meta = _nested_mapping(entry.get("download"))

    source_id = _first_nonempty(
        entry.get("source_id"),
        entry.get("book_id"),
        entry.get("document_id"),
        entry.get("id"),
        entry.get("slug"),
    )
    if not source_id:
        raise ValueError(f"manifest entry {index} lacks source_id/book_id/id")

    slug = _first_nonempty(entry.get("slug"), source_id)
    subject = _first_nonempty(entry.get("subject"), entry.get("ders"))
    source = _first_nonempty(
        entry.get("authority"),
        entry.get("source_name"),
        entry.get("source") if not isinstance(entry.get("source"), Mapping) else None,
        source_meta.get("authority"),
        source_meta.get("name"),
        source_meta.get("id"),
        entry.get("portal"),
        entry.get("publisher"),
        source_id,
    )
    pdf_url = _first_nonempty(
        entry.get("pdf_url"),
        entry.get("download_url"),
        download_meta.get("url"),
        source_meta.get("pdf_url"),
        entry.get("url"),
        entry.get("source_url"),
    )
    pdf_url = _validate_url(pdf_url, source_id=source_id)

    raw_grade = entry.get("grade", entry.get("class", entry.get("sinif")))
    grade: int | None
    if raw_grade is None or str(raw_grade).strip() == "":
        grade = None
    else:
        try:
            grade = int(raw_grade)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"book {source_id!r} has invalid grade {raw_grade!r}"
            ) from exc
        if grade < 1:
            raise ValueError(f"book {source_id!r} has invalid grade {grade}")

    filename_value = _first_nonempty(
        entry.get("filename"),
        entry.get("local_filename"),
        download_meta.get("filename"),
        f"{slug}.pdf",
    )
    filename = _safe_filename(filename_value)
    expected_sha256 = _normalize_expected_sha256(
        entry.get(
            "expected_sha256",
            download_meta.get("expected_sha256", entry.get("sha256")),
        ),
        source_id=source_id,
    )
    return TextbookSource(
        source_id=source_id,
        slug=slug,
        grade=grade,
        subject=subject,
        source=source,
        pdf_url=pdf_url,
        filename=filename,
        expected_sha256=expected_sha256,
        manifest_entry=entry,
    )


def load_manifest(path: Path | str) -> list[TextbookSource]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    books = [
        _parse_book(entry, index=index)
        for index, entry in enumerate(_manifest_entries(payload))
    ]
    seen_ids: set[str] = set()
    seen_filenames: set[str] = set()
    for book in books:
        normalized_id = book.source_id.casefold()
        normalized_filename = book.filename.casefold()
        if normalized_id in seen_ids:
            raise ValueError(f"duplicate source_id: {book.source_id}")
        if normalized_filename in seen_filenames:
            raise ValueError(f"duplicate output filename: {book.filename}")
        seen_ids.add(normalized_id)
        seen_filenames.add(normalized_filename)
    return books


def select_books(
    books: Iterable[TextbookSource],
    *,
    grades: Iterable[int] = (),
    subjects: Iterable[str] = (),
    sources: Iterable[str] = (),
) -> list[TextbookSource]:
    grade_filter = set(grades)
    subject_filter = {
        value.strip().casefold() for value in subjects if value.strip()
    }
    source_filter = {
        value.strip().casefold() for value in sources if value.strip()
    }
    selected: list[TextbookSource] = []
    for book in books:
        if grade_filter and book.grade not in grade_filter:
            continue
        if subject_filter and book.subject.casefold() not in subject_filter:
            continue
        source_values = {book.source.casefold(), book.source_id.casefold()}
        if source_filter and source_values.isdisjoint(source_filter):
            continue
        selected.append(book)
    return selected


def _staging_paths(
    staging_dir: Path,
    book: TextbookSource,
) -> tuple[Path, Path, Path]:
    root = staging_dir.expanduser().resolve()
    final_path = root / book.filename
    provenance_path = root / f"{book.filename}.provenance.json"
    canonical_part = root / f"{book.filename}.part"
    for path in (final_path, provenance_path, canonical_part):
        if path.parent != root:
            raise ValueError(f"output escapes staging directory: {path}")
    return final_path, provenance_path, canonical_part


def _refuse_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing staging target(s): "
            + ", ".join(existing)
        )


def _retryable(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return True
    return int(response.status_code) in _RETRYABLE_STATUS_CODES


def _download_attempt(
    *,
    session: requests.Session,
    book: TextbookSource,
    attempt_path: Path,
    timeout: tuple[float, float],
    chunk_size: int,
) -> tuple[dict[str, Any], str, int]:
    response = session.get(
        book.pdf_url,
        stream=True,
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": "VLM-textbook-stager/1.0"},
    )
    response.raise_for_status()

    digest = hashlib.sha256()
    byte_count = 0
    prefix = bytearray()
    with attempt_path.open("xb") as output:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            output.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
            if len(prefix) < len(_PDF_MAGIC):
                prefix.extend(chunk[: len(_PDF_MAGIC) - len(prefix)])
        output.flush()
        os.fsync(output.fileno())

    if bytes(prefix) != _PDF_MAGIC:
        raise ValueError(
            f"downloaded payload is not a PDF: expected {_PDF_MAGIC!r}"
        )
    if byte_count < len(_PDF_MAGIC):
        raise ValueError("downloaded PDF is empty or truncated")

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.strip().isdigit():
        expected_bytes = int(content_length)
        if expected_bytes != byte_count:
            raise ValueError(
                "Content-Length mismatch: "
                f"header={expected_bytes}, downloaded={byte_count}"
            )

    sha256 = digest.hexdigest()
    if (
        book.expected_sha256 is not None
        and sha256 != book.expected_sha256
    ):
        raise ValueError(
            f"SHA256 mismatch for {book.source_id}: "
            f"expected={book.expected_sha256}, actual={sha256}"
        )
    http = {
        "status_code": int(response.status_code),
        "final_url": str(getattr(response, "url", book.pdf_url)),
        "date": response.headers.get("Date"),
        "last_modified": response.headers.get("Last-Modified"),
        "etag": response.headers.get("ETag"),
        "content_type": response.headers.get("Content-Type"),
        "content_length": response.headers.get("Content-Length"),
    }
    return http, sha256, byte_count


def download_book(
    book: TextbookSource,
    *,
    staging_dir: Path | str,
    session: requests.Session | None = None,
    retries: int = 2,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    backoff_seconds: float = 1.0,
    chunk_size: int = 1024 * 1024,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be non-negative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    staging_root = Path(staging_dir).expanduser().resolve()
    if staging_root.exists() and not staging_root.is_dir():
        raise NotADirectoryError(f"staging path is not a directory: {staging_root}")
    staging_root.mkdir(parents=True, exist_ok=True)

    final_path, provenance_path, canonical_part = _staging_paths(
        staging_root,
        book,
    )
    _refuse_existing((final_path, provenance_path, canonical_part))
    active_session = session or requests.Session()

    last_error: BaseException | None = None
    http: dict[str, Any] | None = None
    sha256 = ""
    byte_count = 0
    successful_attempt_path: Path | None = None
    for attempt in range(1, retries + 2):
        attempt_path = staging_root / (
            f"{book.filename}.part.attempt-{attempt}-{uuid4().hex}"
        )
        try:
            http, sha256, byte_count = _download_attempt(
                session=active_session,
                book=book,
                attempt_path=attempt_path,
                timeout=(connect_timeout, read_timeout),
                chunk_size=chunk_size,
            )
            successful_attempt_path = attempt_path
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt > retries or not _retryable(exc):
                raise
            if backoff_seconds:
                sleep(backoff_seconds * (2 ** (attempt - 1)))
        except OSError as exc:
            last_error = exc
            if attempt > retries:
                raise
            if backoff_seconds:
                sleep(backoff_seconds * (2 ** (attempt - 1)))

    if successful_attempt_path is None or http is None:
        raise RuntimeError("download failed without a successful attempt") from last_error

    downloaded_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": 1,
        "status": "downloaded",
        "source_id": book.source_id,
        "slug": book.slug,
        "grade": book.grade,
        "subject": book.subject,
        "source": book.source,
        "pdf_url": book.pdf_url,
        "filename": book.filename,
        "staged_path": str(final_path),
        "sha256": sha256,
        "expected_sha256": book.expected_sha256,
        "sha256_verified": (
            book.expected_sha256 is not None
            and sha256 == book.expected_sha256
        ),
        "bytes": byte_count,
        "downloaded_at": downloaded_at,
        "http_timestamp": http.get("date"),
        "http_last_modified": http.get("last_modified"),
        "http": http,
    }

    provenance_part = staging_root / (
        f"{book.filename}.provenance.json.part"
    )
    _refuse_existing(
        (final_path, provenance_path, canonical_part, provenance_part)
    )
    with provenance_part.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())

    _refuse_existing((final_path, provenance_path, canonical_part))
    successful_attempt_path.rename(canonical_part)
    _refuse_existing((final_path, provenance_path))
    canonical_part.rename(final_path)
    provenance_part.rename(provenance_path)
    return provenance


def _book_plan(book: TextbookSource, staging_dir: Path) -> dict[str, Any]:
    final_path, provenance_path, part_path = _staging_paths(staging_dir, book)
    return {
        "source_id": book.source_id,
        "slug": book.slug,
        "grade": book.grade,
        "subject": book.subject,
        "source": book.source,
        "pdf_url": book.pdf_url,
        "filename": book.filename,
        "expected_sha256": book.expected_sha256,
        "target": str(final_path),
        "provenance": str(provenance_path),
        "exists": any(
            path.exists() for path in (final_path, provenance_path, part_path)
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage official textbook PDFs from a manifest. Downloads are "
            "atomic, non-overwriting, and never invoke the import adapter."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--grade", type=int, action="append", default=[])
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="explicitly allow an unfiltered download of every manifest entry",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="additional attempts after the first request",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    books = load_manifest(args.manifest)
    selected = select_books(
        books,
        grades=args.grade,
        subjects=args.subject,
        sources=args.source,
    )
    if not selected:
        raise SystemExit("No textbook entries matched the requested filters")

    staging_dir = args.staging_dir.expanduser().resolve()
    plan = [_book_plan(book, staging_dir) for book in selected]
    if args.list or args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "list" if args.list else "dry-run",
                    "manifest": str(args.manifest),
                    "staging_dir": str(staging_dir),
                    "selected": len(plan),
                    "books": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    has_filters = bool(args.grade or args.subject or args.source)
    if not has_filters and not args.all:
        raise SystemExit(
            "Refusing an unfiltered download. Add a grade/subject/source "
            "filter or pass --all explicitly."
        )

    session = requests.Session()
    results = [
        download_book(
            book,
            staging_dir=staging_dir,
            session=session,
            retries=args.retries,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            backoff_seconds=args.backoff_seconds,
        )
        for book in selected
    ]
    print(
        json.dumps(
            {
                "mode": "download",
                "manifest": str(args.manifest),
                "staging_dir": str(staging_dir),
                "downloaded": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

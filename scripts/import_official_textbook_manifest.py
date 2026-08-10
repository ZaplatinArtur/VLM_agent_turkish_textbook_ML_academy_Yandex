"""Safely import staged official textbook PDFs declared by a manifest.

The downloader and the PDF adapter remain deliberately separate:

* this script never downloads a PDF;
* staged inputs must be named ``<source_id>.pdf`` and accompanied by the
  downloader's ``<source_id>.pdf.provenance.json``;
* every selected input and every existing output is preflighted before the
  first import starts;
* imports are executed sequentially through
  :func:`import_official_textbook_pdf`;
* existing outputs are never overwritten.  ``--resume`` only skips an entry
  after independently validating its report and published artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schemas.retrieve import RetrievedChunk
from scripts import import_official_textbook_pdf as pdf_adapter


_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ManifestTextbook:
    source_id: str
    staging_slug: str
    slug: str
    grade: int
    subject: str
    publisher: str
    edition: str
    source_url: str
    expected_sha256: str | None
    manifest_entry: Mapping[str, Any]


@dataclass(frozen=True)
class StagedTextbook:
    book: ManifestTextbook
    pdf_path: Path
    provenance_path: Path
    pdf_sha256: str
    pdf_bytes: int
    retrieved_at: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ImportTargets:
    images_dir: Path
    chunks_jsonl: Path
    report_path: Path


@dataclass(frozen=True)
class ImportPlan:
    staged: StagedTextbook
    targets: ImportTargets
    action: str
    validated_report: Mapping[str, Any] | None = None


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _manifest_entries(
    payload: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if isinstance(payload, list):
        manifest: Mapping[str, Any] = {}
        raw_entries = payload
    elif isinstance(payload, Mapping):
        manifest = payload
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
    return manifest, entries


def _validate_source_id(value: Any, *, index: int) -> str:
    source_id = _first_nonempty(value)
    if (
        not source_id
        or Path(source_id).name != source_id
        or source_id in {".", ".."}
        or _SOURCE_ID_RE.fullmatch(source_id) is None
    ):
        raise ValueError(
            f"manifest entry {index} has unsafe source_id {source_id!r}"
        )
    return source_id


def _output_slug(entry: Mapping[str, Any], *, source_id: str) -> str:
    explicit = _first_nonempty(entry.get("import_slug"), entry.get("slug"))
    if explicit:
        if _SLUG_RE.fullmatch(explicit) is None:
            raise ValueError(
                f"book {source_id!r} has invalid import slug {explicit!r}"
            )
        return explicit

    slug = re.sub(r"[^a-z0-9]+", "-", source_id.casefold()).strip("-")
    if not slug or _SLUG_RE.fullmatch(slug) is None:
        raise ValueError(
            f"book {source_id!r} cannot be converted to a safe import slug"
        )
    return slug


def _positive_grade(value: Any, *, source_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"book {source_id!r} has invalid grade {value!r}")
    try:
        grade = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"book {source_id!r} has invalid grade {value!r}"
        ) from exc
    if grade < 1:
        raise ValueError(f"book {source_id!r} has invalid grade {grade}")
    return grade


def _absolute_http_url(value: Any, *, source_id: str) -> str:
    url = _first_nonempty(value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"book {source_id!r} has no absolute HTTP(S) PDF URL"
        )
    return url


def _expected_sha256(entry: Mapping[str, Any], *, source_id: str) -> str | None:
    download = entry.get("download")
    download_meta = download if isinstance(download, Mapping) else {}
    value = entry.get(
        "expected_sha256",
        download_meta.get("expected_sha256", entry.get("sha256")),
    )
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"book {source_id!r} has invalid expected_sha256"
        )
    return normalized


def load_manifest(path: Path | str) -> list[ManifestTextbook]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest, raw_entries = _manifest_entries(payload)
    manifest_edition = _first_nonempty(
        manifest.get("academic_year"),
        manifest.get("edition"),
        manifest.get("catalog_snapshot"),
    )

    books: list[ManifestTextbook] = []
    for index, entry in enumerate(raw_entries):
        source_id = _validate_source_id(entry.get("source_id"), index=index)
        staging_slug = _first_nonempty(entry.get("slug"), source_id)
        subject = _first_nonempty(entry.get("subject"))
        publisher = _first_nonempty(
            entry.get("publisher"),
            entry.get("portal"),
        )
        edition = _first_nonempty(
            entry.get("edition"),
            manifest_edition,
            entry.get("curriculum"),
        )
        if not subject:
            raise ValueError(f"book {source_id!r} has no subject")
        if not publisher:
            raise ValueError(f"book {source_id!r} has no publisher")
        if not edition:
            raise ValueError(f"book {source_id!r} has no edition")

        books.append(
            ManifestTextbook(
                source_id=source_id,
                staging_slug=staging_slug,
                slug=_output_slug(entry, source_id=source_id),
                grade=_positive_grade(entry.get("grade"), source_id=source_id),
                subject=subject,
                publisher=publisher,
                edition=edition,
                source_url=_absolute_http_url(
                    entry.get("pdf_url", entry.get("source_url")),
                    source_id=source_id,
                ),
                expected_sha256=_expected_sha256(
                    entry,
                    source_id=source_id,
                ),
                manifest_entry=entry,
            )
        )

    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for book in books:
        normalized_id = book.source_id.casefold()
        if normalized_id in seen_ids:
            raise ValueError(f"duplicate source_id: {book.source_id}")
        if book.slug in seen_slugs:
            raise ValueError(f"duplicate import slug: {book.slug}")
        seen_ids.add(normalized_id)
        seen_slugs.add(book.slug)
    return books


def select_books(
    books: Iterable[ManifestTextbook],
    *,
    grades: Iterable[int] = (),
    subjects: Iterable[str] = (),
    exclude_subjects: Iterable[str] = (),
) -> list[ManifestTextbook]:
    grade_filter = set(grades)
    subject_filter = {
        str(value).strip().casefold()
        for value in subjects
        if str(value).strip()
    }
    excluded = {
        str(value).strip().casefold()
        for value in exclude_subjects
        if str(value).strip()
    }
    selected: list[ManifestTextbook] = []
    for book in books:
        normalized_subject = book.subject.casefold()
        if grade_filter and book.grade not in grade_filter:
            continue
        if subject_filter and normalized_subject not in subject_filter:
            continue
        if normalized_subject in excluded:
            continue
        selected.append(book)
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_iso_timestamp(value: Any, *, source_id: str) -> str:
    timestamp = _first_nonempty(value)
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"provenance for {source_id!r} has invalid downloaded_at"
        ) from exc
    return timestamp


def _staged_paths(
    staging_dir: Path,
    book: ManifestTextbook,
) -> tuple[Path, Path]:
    root = staging_dir.expanduser().resolve()
    pdf_path = root / f"{book.source_id}.pdf"
    provenance_path = root / f"{book.source_id}.pdf.provenance.json"
    if pdf_path.parent != root or provenance_path.parent != root:
        raise ValueError(
            f"staged paths for {book.source_id!r} escape staging directory"
        )
    return pdf_path, provenance_path


def validate_staged_textbook(
    staging_dir: Path | str,
    book: ManifestTextbook,
) -> StagedTextbook:
    pdf_path, provenance_path = _staged_paths(Path(staging_dir), book)
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"staged PDF does not exist for {book.source_id}: {pdf_path}"
        )
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "staged provenance does not exist for "
            f"{book.source_id}: {provenance_path}"
        )
    with pdf_path.open("rb") as handle:
        if handle.read(len(_PDF_MAGIC)) != _PDF_MAGIC:
            raise ValueError(
                f"staged input is not a PDF for {book.source_id}: {pdf_path}"
            )

    provenance = _mapping(
        json.loads(provenance_path.read_text(encoding="utf-8")),
        label=f"provenance for {book.source_id}",
    )
    if provenance.get("schema_version") != 1:
        raise ValueError(
            f"provenance for {book.source_id!r} has unsupported schema_version"
        )
    if provenance.get("status") != "downloaded":
        raise ValueError(
            f"provenance for {book.source_id!r} is not downloaded"
        )
    if provenance.get("source_id") != book.source_id:
        raise ValueError(
            f"provenance source_id mismatch for {book.source_id!r}"
        )
    if provenance.get("filename") != pdf_path.name:
        raise ValueError(
            f"provenance filename mismatch for {book.source_id!r}"
        )
    if _first_nonempty(provenance.get("slug")) != book.staging_slug:
        raise ValueError(f"provenance slug mismatch for {book.source_id!r}")
    if provenance.get("grade") != book.grade:
        raise ValueError(f"provenance grade mismatch for {book.source_id!r}")
    if (
        _first_nonempty(provenance.get("subject")).casefold()
        != book.subject.casefold()
    ):
        raise ValueError(f"provenance subject mismatch for {book.source_id!r}")
    if provenance.get("pdf_url") != book.source_url:
        raise ValueError(f"provenance pdf_url mismatch for {book.source_id!r}")

    staged_path_value = _first_nonempty(provenance.get("staged_path"))
    if staged_path_value and Path(staged_path_value).name != pdf_path.name:
        raise ValueError(
            f"provenance staged_path mismatch for {book.source_id!r}"
        )

    actual_bytes = pdf_path.stat().st_size
    reported_bytes = provenance.get("bytes")
    if (
        isinstance(reported_bytes, bool)
        or not isinstance(reported_bytes, int)
        or reported_bytes != actual_bytes
    ):
        raise ValueError(
            f"provenance byte count mismatch for {book.source_id!r}"
        )
    actual_sha256 = _sha256_file(pdf_path)
    reported_sha256 = _first_nonempty(provenance.get("sha256")).casefold()
    if (
        _SHA256_RE.fullmatch(reported_sha256) is None
        or reported_sha256 != actual_sha256
    ):
        raise ValueError(f"provenance SHA256 mismatch for {book.source_id!r}")
    if (
        book.expected_sha256 is not None
        and actual_sha256 != book.expected_sha256
    ):
        raise ValueError(
            f"manifest SHA256 mismatch for {book.source_id!r}: "
            f"expected={book.expected_sha256}, actual={actual_sha256}"
        )

    return StagedTextbook(
        book=book,
        pdf_path=pdf_path,
        provenance_path=provenance_path,
        pdf_sha256=actual_sha256,
        pdf_bytes=actual_bytes,
        retrieved_at=_validate_iso_timestamp(
            provenance.get("downloaded_at"),
            source_id=book.source_id,
        ),
        provenance=provenance,
    )


def _targets(data_dir: Path | str, slug: str) -> ImportTargets:
    root = Path(data_dir).expanduser().resolve()
    return ImportTargets(
        images_dir=root / "books" / slug,
        chunks_jsonl=root / "chunks" / "jsonl" / f"{slug}.jsonl",
        report_path=(
            root
            / "reports"
            / "official_textbook_pdf"
            / f"{slug}.json"
        ),
    )


def _assert_report_path(
    value: Any,
    *,
    expected: Path,
    label: str,
) -> None:
    rendered = _first_nonempty(value)
    if not rendered or Path(rendered).expanduser().resolve() != expected:
        raise ValueError(f"resume report {label} does not match {expected}")


def _read_chunks(path: Path, *, source_id: str) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunks.append(RetrievedChunk.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"resume JSONL for {source_id!r} is invalid at "
                    f"line {line_number}"
                ) from exc
    return chunks


def validate_existing_report(plan: ImportPlan) -> Mapping[str, Any]:
    staged = plan.staged
    book = staged.book
    targets = plan.targets
    report = _mapping(
        json.loads(targets.report_path.read_text(encoding="utf-8")),
        label=f"resume report for {book.source_id}",
    )
    if report.get("schema_version") != 1:
        raise ValueError(
            f"resume report for {book.source_id!r} has unsupported schema"
        )
    if report.get("status") != "validated":
        raise ValueError(
            f"resume report for {book.source_id!r} is not validated"
        )
    if report.get("ingest_method") != pdf_adapter.INGEST_METHOD:
        raise ValueError(
            f"resume report ingest method mismatch for {book.source_id!r}"
        )

    source = _mapping(
        report.get("source"),
        label=f"resume report source for {book.source_id}",
    )
    if source.get("source_url") != book.source_url:
        raise ValueError(
            f"resume report source URL mismatch for {book.source_id!r}"
        )
    if source.get("pdf_sha256") != staged.pdf_sha256:
        raise ValueError(
            f"resume report PDF SHA256 mismatch for {book.source_id!r}"
        )
    if source.get("retrieved_at") != staged.retrieved_at:
        raise ValueError(
            f"resume report retrieval timestamp mismatch for {book.source_id!r}"
        )
    if Path(_first_nonempty(source.get("pdf_path"))).name != staged.pdf_path.name:
        raise ValueError(
            f"resume report PDF filename mismatch for {book.source_id!r}"
        )
    page_count = source.get("page_count")
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
    ):
        raise ValueError(
            f"resume report page count is invalid for {book.source_id!r}"
        )

    reported_book = _mapping(
        report.get("book"),
        label=f"resume report book for {book.source_id}",
    )
    expected_book = {
        "slug": book.slug,
        "grade": book.grade,
        "subject": book.subject,
        "publisher": book.publisher,
        "edition": book.edition,
    }
    for key, expected in expected_book.items():
        if reported_book.get(key) != expected:
            raise ValueError(
                f"resume report book.{key} mismatch for {book.source_id!r}"
            )

    outputs = _mapping(
        report.get("outputs"),
        label=f"resume report outputs for {book.source_id}",
    )
    _assert_report_path(
        outputs.get("images_dir"),
        expected=targets.images_dir,
        label="images_dir",
    )
    _assert_report_path(
        outputs.get("report_path"),
        expected=targets.report_path,
        label="report_path",
    )
    chunks_report = _mapping(
        outputs.get("chunks_jsonl"),
        label=f"resume report chunks_jsonl for {book.source_id}",
    )
    _assert_report_path(
        chunks_report.get("path"),
        expected=targets.chunks_jsonl,
        label="chunks_jsonl.path",
    )

    validation = _mapping(
        report.get("validation"),
        label=f"resume report validation for {book.source_id}",
    )
    required_true = (
        "counts_match",
        "text_threshold_passed",
        "retrieved_chunk_validation_passed",
    )
    if any(validation.get(key) is not True for key in required_true):
        raise ValueError(
            f"resume report validation flags failed for {book.source_id!r}"
        )
    count_fields = (
        "pdf_page_count",
        "rendered_image_count",
        "jsonl_record_count",
    )
    if any(validation.get(key) != page_count for key in count_fields):
        raise ValueError(
            f"resume report validation counts mismatch for {book.source_id!r}"
        )

    if not targets.images_dir.is_dir() or not targets.chunks_jsonl.is_file():
        raise ValueError(
            f"resume outputs are incomplete for {book.source_id!r}"
        )
    expected_image_names = [
        f"{page:04d}.jpg" for page in range(1, page_count + 1)
    ]
    actual_entries = sorted(path.name for path in targets.images_dir.iterdir())
    if actual_entries != expected_image_names:
        raise ValueError(
            f"resume image sequence mismatch for {book.source_id!r}"
        )

    image_reports = outputs.get("images")
    if not isinstance(image_reports, list) or len(image_reports) != page_count:
        raise ValueError(
            f"resume image report count mismatch for {book.source_id!r}"
        )
    image_hashes: dict[int, str] = {}
    for page_number, image_report in enumerate(image_reports, 1):
        record = _mapping(
            image_report,
            label=f"resume image report for {book.source_id}",
        )
        image_path = targets.images_dir / f"{page_number:04d}.jpg"
        expected_relative = f"books/{book.slug}/{page_number:04d}.jpg"
        actual_hash = _sha256_file(image_path)
        if (
            record.get("page") != page_number
            or record.get("path") != expected_relative
            or record.get("sha256") != actual_hash
            or record.get("bytes") != image_path.stat().st_size
        ):
            raise ValueError(
                f"resume image report mismatch for {book.source_id!r} "
                f"page {page_number}"
            )
        image_hashes[page_number] = actual_hash

    actual_jsonl_sha256 = _sha256_file(targets.chunks_jsonl)
    chunks = _read_chunks(
        targets.chunks_jsonl,
        source_id=book.source_id,
    )
    if (
        chunks_report.get("sha256") != actual_jsonl_sha256
        or chunks_report.get("bytes") != targets.chunks_jsonl.stat().st_size
        or chunks_report.get("records") != page_count
        or len(chunks) != page_count
    ):
        raise ValueError(
            f"resume JSONL report mismatch for {book.source_id!r}"
        )

    for page_number, chunk in enumerate(chunks, 1):
        expected_chunk_id = f"{book.slug}:{page_number:04d}"
        expected_image = f"books/{book.slug}/{page_number:04d}.jpg"
        metadata = chunk.metadata
        if chunk.chunk_id != expected_chunk_id:
            raise ValueError(
                f"resume chunk id mismatch for {book.source_id!r} "
                f"page {page_number}"
            )
        expected_metadata = {
            "textbook": book.slug,
            "page": page_number,
            "page_count": page_count,
            "grade": book.grade,
            "subject": book.subject,
            "publisher": book.publisher,
            "edition": book.edition,
            "source_url": book.source_url,
            "source_pdf_filename": staged.pdf_path.name,
            "source_pdf_sha256": staged.pdf_sha256,
            "source_retrieved_at": staged.retrieved_at,
            "image_sha256": image_hashes[page_number],
            "ingest_method": pdf_adapter.INGEST_METHOD,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(
                f"resume chunk metadata mismatch for {book.source_id!r} "
                f"page {page_number}"
            )
        if (
            len(chunk.images) != 1
            or chunk.images[0].image_id != expected_chunk_id
            or chunk.images[0].format != "file_path"
            or chunk.images[0].mime_type != "image/jpeg"
            or chunk.images[0].data != expected_image
        ):
            raise ValueError(
                f"resume chunk image mismatch for {book.source_id!r} "
                f"page {page_number}"
            )
    return report


def _preflight_plan(
    *,
    book: ManifestTextbook,
    staging_dir: Path,
    data_dir: Path,
    resume: bool,
) -> ImportPlan:
    staged = validate_staged_textbook(staging_dir, book)
    targets = _targets(data_dir, book.slug)
    target_paths = (
        targets.images_dir,
        targets.chunks_jsonl,
        targets.report_path,
    )
    existing = [path.exists() for path in target_paths]
    if any(existing):
        if not resume:
            rendered = ", ".join(
                str(path)
                for path, exists in zip(target_paths, existing)
                if exists
            )
            raise FileExistsError(
                "refusing to overwrite existing import target(s): " + rendered
            )
        if not all(existing):
            raise FileExistsError(
                "cannot resume an incomplete output set for "
                f"{book.source_id}: "
                + ", ".join(
                    f"{path}={'present' if exists else 'missing'}"
                    for path, exists in zip(target_paths, existing)
                )
            )
        provisional = ImportPlan(
            staged=staged,
            targets=targets,
            action="resume",
        )
        report = validate_existing_report(provisional)
        return ImportPlan(
            staged=staged,
            targets=targets,
            action="resume",
            validated_report=report,
        )
    return ImportPlan(
        staged=staged,
        targets=targets,
        action="import",
    )


def _plan_record(plan: ImportPlan) -> dict[str, Any]:
    book = plan.staged.book
    return {
        "source_id": book.source_id,
        "slug": book.slug,
        "grade": book.grade,
        "subject": book.subject,
        "action": plan.action,
        "pdf": str(plan.staged.pdf_path),
        "provenance": str(plan.staged.provenance_path),
        "pdf_sha256": plan.staged.pdf_sha256,
        "images_dir": str(plan.targets.images_dir),
        "chunks_jsonl": str(plan.targets.chunks_jsonl),
        "report": str(plan.targets.report_path),
    }


def import_official_textbook_manifest(
    *,
    manifest_path: Path | str,
    staging_dir: Path | str,
    data_dir: Path | str,
    grades: Iterable[int] = (),
    subjects: Iterable[str] = (),
    exclude_subjects: Iterable[str] = (),
    dry_run: bool = False,
    resume: bool = False,
    min_text_page_ratio: float = pdf_adapter.DEFAULT_MIN_TEXT_PAGE_RATIO,
    render_dpi: int = pdf_adapter.DEFAULT_RENDER_DPI,
    jpeg_quality: int = pdf_adapter.DEFAULT_JPEG_QUALITY,
    import_one: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Preflight and sequentially import selected staged textbooks."""

    grades = tuple(grades)
    subjects = tuple(subjects)
    exclude_subjects = tuple(exclude_subjects)
    if not 0.0 <= min_text_page_ratio <= 1.0:
        raise ValueError("min_text_page_ratio must be between 0 and 1")
    if isinstance(render_dpi, bool) or render_dpi < 36:
        raise ValueError("render_dpi must be an integer of at least 36")
    if (
        isinstance(jpeg_quality, bool)
        or not 1 <= jpeg_quality <= 100
    ):
        raise ValueError("jpeg_quality must be an integer between 1 and 100")

    manifest = Path(manifest_path).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    data = Path(data_dir).expanduser().resolve()
    selected = select_books(
        load_manifest(manifest),
        grades=grades,
        subjects=subjects,
        exclude_subjects=exclude_subjects,
    )
    if not selected:
        raise ValueError("no textbook entries matched the requested filters")

    # Preflight the whole selection before the first adapter invocation.  This
    # prevents a missing provenance file or invalid resume report late in the
    # manifest from causing an avoidable partially imported batch.
    plans = [
        _preflight_plan(
            book=book,
            staging_dir=staging,
            data_dir=data,
            resume=resume,
        )
        for book in selected
    ]
    base_result: dict[str, Any] = {
        "schema_version": 1,
        "mode": "dry-run" if dry_run else "import",
        "manifest": str(manifest),
        "staging_dir": str(staging),
        "data_dir": str(data),
        "resume": bool(resume),
        "filters": {
            "grades": list(grades),
            "subjects": list(subjects),
            "exclude_subjects": list(exclude_subjects),
        },
        "selected": len(plans),
        "planned_imports": sum(plan.action == "import" for plan in plans),
        "validated_resumes": sum(plan.action == "resume" for plan in plans),
    }
    if dry_run:
        return {
            **base_result,
            "books": [_plan_record(plan) for plan in plans],
        }

    importer = import_one or pdf_adapter.import_official_textbook_pdf
    results: list[dict[str, Any]] = []
    for plan in plans:
        book = plan.staged.book
        if plan.action == "resume":
            report = plan.validated_report or {}
            results.append(
                {
                    **_plan_record(plan),
                    "status": "resumed_validated",
                    "page_count": (
                        report.get("source", {}).get("page_count")
                        if isinstance(report.get("source"), Mapping)
                        else None
                    ),
                }
            )
            continue

        importer(
            pdf_path=plan.staged.pdf_path,
            data_dir=data,
            slug=book.slug,
            source_url=book.source_url,
            grade=book.grade,
            subject=book.subject,
            publisher=book.publisher,
            edition=book.edition,
            retrieved_at=plan.staged.retrieved_at,
            report_path=plan.targets.report_path,
            min_text_page_ratio=min_text_page_ratio,
            render_dpi=render_dpi,
            jpeg_quality=jpeg_quality,
        )
        published = ImportPlan(
            staged=plan.staged,
            targets=plan.targets,
            action="resume",
        )
        validated = validate_existing_report(published)
        results.append(
            {
                **_plan_record(plan),
                "status": "imported_validated",
                "page_count": validated["source"]["page_count"],
            }
        )

    return {
        **base_result,
        "imported": sum(item["status"] == "imported_validated" for item in results),
        "resumed": sum(item["status"] == "resumed_validated" for item in results),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially import staged official textbook PDFs from a "
            "manifest. Existing outputs are never overwritten; --resume "
            "requires a fully validated prior report."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--grade", type=int, action="append", default=[])
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--exclude-subject", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-text-page-ratio",
        type=float,
        default=pdf_adapter.DEFAULT_MIN_TEXT_PAGE_RATIO,
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=pdf_adapter.DEFAULT_RENDER_DPI,
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=pdf_adapter.DEFAULT_JPEG_QUALITY,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = import_official_textbook_manifest(
        manifest_path=args.manifest,
        staging_dir=args.staging_dir,
        data_dir=args.data_dir,
        grades=args.grade,
        subjects=args.subject,
        exclude_subjects=args.exclude_subject,
        dry_run=args.dry_run,
        resume=args.resume,
        min_text_page_ratio=args.min_text_page_ratio,
        render_dpi=args.render_dpi,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

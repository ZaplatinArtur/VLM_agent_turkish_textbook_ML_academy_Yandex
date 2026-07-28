"""Import one authorized official textbook PDF into the page-chunk corpus.

The adapter is intentionally metadata-driven and non-destructive. It does not
download a source PDF and it never overwrites an existing image directory,
JSONL file, or report. The caller is responsible for confirming that the PDF
may be processed under the source's copyright and usage terms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schemas.media import ImageRef
from schemas.retrieve import RetrievedChunk


INGEST_METHOD = "official_textbook_pdf_native_text_v1"
DEFAULT_MIN_TEXT_PAGE_RATIO = 0.80
DEFAULT_RENDER_DPI = 150
DEFAULT_JPEG_QUALITY = 90
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _load_fitz() -> Any:
    try:
        return importlib.import_module("fitz")
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to import official textbook PDFs. "
            "Install it with `python -m pip install PyMuPDF` and retry."
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_nonempty(name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    return cleaned


def _validate_source_url(value: str) -> str:
    source_url = _validate_nonempty("source_url", value)
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    return source_url


def _validate_retrieved_at(value: str) -> str:
    retrieved_at = _validate_nonempty("retrieved_at", value)
    normalized = retrieved_at[:-1] + "+00:00" if retrieved_at.endswith("Z") else retrieved_at
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "retrieved_at must be an ISO-8601 date or datetime"
        ) from exc
    return retrieved_at


def _validate_inputs(
    *,
    pdf_path: Path,
    slug: str,
    source_url: str,
    grade: int,
    subject: str,
    publisher: str,
    edition: str,
    retrieved_at: str,
    min_text_page_ratio: float,
    render_dpi: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF input does not exist: {pdf_path}")
    if pdf_path.suffix.casefold() != ".pdf":
        raise ValueError(f"PDF input must have a .pdf suffix: {pdf_path}")

    slug = _validate_nonempty("slug", slug)
    if _SLUG_RE.fullmatch(slug) is None:
        raise ValueError(
            "slug must contain only lowercase ASCII letters, digits, and "
            "single-component hyphens"
        )
    if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
        raise ValueError("grade must be a positive integer")
    if not 0.0 <= min_text_page_ratio <= 1.0:
        raise ValueError("min_text_page_ratio must be between 0 and 1")
    if isinstance(render_dpi, bool) or render_dpi < 36:
        raise ValueError("render_dpi must be an integer of at least 36")
    if isinstance(jpeg_quality, bool) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer between 1 and 100")

    return {
        "pdf_path": pdf_path,
        "slug": slug,
        "source_url": _validate_source_url(source_url),
        "grade": grade,
        "subject": _validate_nonempty("subject", subject),
        "publisher": _validate_nonempty("publisher", publisher),
        "edition": _validate_nonempty("edition", edition),
        "retrieved_at": _validate_retrieved_at(retrieved_at),
        "min_text_page_ratio": float(min_text_page_ratio),
        "render_dpi": int(render_dpi),
        "jpeg_quality": int(jpeg_quality),
    }


def _output_paths(
    *,
    data_dir: Path,
    slug: str,
    report_path: Path | None,
) -> tuple[Path, Path, Path]:
    images_dir = data_dir / "books" / slug
    jsonl_path = data_dir / "chunks" / "jsonl" / f"{slug}.jsonl"
    final_report_path = report_path or (
        data_dir
        / "reports"
        / "official_textbook_pdf"
        / f"{slug}.json"
    )
    return images_dir, jsonl_path, final_report_path


def _refuse_existing_targets(targets: list[Path]) -> None:
    existing = [path for path in targets if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "refusing to overwrite existing output target(s): " + rendered
        )


def _fitz_version(fitz: Any) -> str:
    version = getattr(fitz, "VersionBind", None)
    if version:
        return str(version)
    versions = getattr(fitz, "version", None)
    if isinstance(versions, (tuple, list)) and versions:
        return str(versions[0])
    return "unknown"


def _jpeg_bytes(
    pixmap: Any,
    *,
    jpeg_quality: int,
) -> tuple[bytes, bool]:
    try:
        return (
            pixmap.tobytes("jpeg", jpg_quality=jpeg_quality),
            True,
        )
    except TypeError:
        # Older PyMuPDF versions can encode JPEG but do not expose quality.
        return pixmap.tobytes("jpeg"), False


def _validate_staged_output(
    *,
    images_dir: Path,
    jsonl_path: Path,
    expected_pages: int,
) -> list[RetrievedChunk]:
    images = sorted(images_dir.glob("*.jpg"))
    expected_names = [f"{page:04d}.jpg" for page in range(1, expected_pages + 1)]
    actual_names = [path.name for path in images]
    if actual_names != expected_names:
        raise ValueError(
            "rendered image sequence mismatch: "
            f"expected={expected_names}, actual={actual_names}"
        )

    chunks: list[RetrievedChunk] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunks.append(RetrievedChunk.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid RetrievedChunk JSON at line {line_number}"
                ) from exc

    if not (expected_pages == len(images) == len(chunks)):
        raise ValueError(
            "page count validation failed: "
            f"pdf={expected_pages}, images={len(images)}, "
            f"jsonl={len(chunks)}"
        )
    for page_number, chunk in enumerate(chunks, 1):
        expected_chunk_id = (
            f"{chunk.metadata.get('textbook')}:{page_number:04d}"
        )
        if chunk.chunk_id != expected_chunk_id:
            raise ValueError(
                f"unexpected chunk id at page {page_number}: "
                f"{chunk.chunk_id!r}"
            )
        if len(chunk.images) != 1:
            raise ValueError(
                f"page {page_number} must have exactly one ImageRef"
            )
        image = chunk.images[0]
        if image.format != "file_path" or image.mime_type != "image/jpeg":
            raise ValueError(
                f"page {page_number} has an incompatible ImageRef"
            )
    return chunks


def _publish_staged_outputs(
    *,
    staged_images_dir: Path,
    staged_jsonl_path: Path,
    staged_report_path: Path,
    images_dir: Path,
    jsonl_path: Path,
    report_path: Path,
) -> None:
    targets = [images_dir, jsonl_path, report_path]
    _refuse_existing_targets(targets)
    images_dir.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _refuse_existing_targets(targets)

    # Each source is renamed only to a previously verified absent target.
    # There is deliberately no replace/overwrite mode.
    staged_images_dir.rename(images_dir)
    staged_jsonl_path.rename(jsonl_path)
    staged_report_path.rename(report_path)


def import_official_textbook_pdf(
    *,
    pdf_path: Path,
    data_dir: Path,
    slug: str,
    source_url: str,
    grade: int,
    subject: str,
    publisher: str,
    edition: str,
    retrieved_at: str,
    report_path: Path | None = None,
    min_text_page_ratio: float = DEFAULT_MIN_TEXT_PAGE_RATIO,
    render_dpi: int = DEFAULT_RENDER_DPI,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Import one local PDF after validating its native text layer.

    All output is prepared and validated in a temporary staging directory.
    Existing final targets cause an error before the PDF is opened.
    """

    validated = _validate_inputs(
        pdf_path=Path(pdf_path),
        slug=slug,
        source_url=source_url,
        grade=grade,
        subject=subject,
        publisher=publisher,
        edition=edition,
        retrieved_at=retrieved_at,
        min_text_page_ratio=min_text_page_ratio,
        render_dpi=render_dpi,
        jpeg_quality=jpeg_quality,
    )
    pdf_path = validated["pdf_path"]
    slug = validated["slug"]
    source_url = validated["source_url"]
    grade = validated["grade"]
    subject = validated["subject"]
    publisher = validated["publisher"]
    edition = validated["edition"]
    retrieved_at = validated["retrieved_at"]
    min_text_page_ratio = validated["min_text_page_ratio"]
    render_dpi = validated["render_dpi"]
    jpeg_quality = validated["jpeg_quality"]

    data_dir = Path(data_dir).expanduser().resolve()
    resolved_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else None
    )
    images_dir, jsonl_path, final_report_path = _output_paths(
        data_dir=data_dir,
        slug=slug,
        report_path=resolved_report_path,
    )
    final_targets = [images_dir, jsonl_path, final_report_path]
    _refuse_existing_targets(final_targets)

    fitz = _load_fitz()
    fitz_version = _fitz_version(fitz)
    pdf_sha256 = _sha256_file(pdf_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(pdf_path)) as document:
        if getattr(document, "needs_pass", False):
            raise ValueError("encrypted PDF requires a password")
        page_count = int(document.page_count)
        if page_count < 1:
            raise ValueError("PDF contains no pages")

        texts = [
            document.load_page(index).get_text("text", sort=True).strip()
            for index in range(page_count)
        ]
        text_page_count = sum(bool(text) for text in texts)
        text_page_ratio = text_page_count / page_count
        required_text_pages = math.ceil(page_count * min_text_page_ratio)
        if text_page_count < required_text_pages:
            raise ValueError(
                "native text page ratio is below threshold: "
                f"text_pages={text_page_count}, pages={page_count}, "
                f"ratio={text_page_ratio:.4f}, "
                f"required_ratio={min_text_page_ratio:.4f}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f".official-pdf-{slug}-",
            dir=data_dir,
        ) as staging:
            staging_dir = Path(staging)
            staged_images_dir = staging_dir / "images"
            staged_images_dir.mkdir()
            staged_jsonl_path = staging_dir / f"{slug}.jsonl"
            staged_report_path = staging_dir / f"{slug}.report.json"

            chunks: list[RetrievedChunk] = []
            image_records: list[dict[str, Any]] = []
            quality_supported_for_all_pages = True
            matrix = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)

            for index, text in enumerate(texts):
                page_number = index + 1
                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                encoded, quality_supported = _jpeg_bytes(
                    pixmap,
                    jpeg_quality=jpeg_quality,
                )
                quality_supported_for_all_pages &= quality_supported
                image_name = f"{page_number:04d}.jpg"
                staged_image_path = staged_images_dir / image_name
                staged_image_path.write_bytes(encoded)
                image_sha256 = _sha256_bytes(encoded)
                text_sha256 = _sha256_bytes(text.encode("utf-8"))
                chunk_id = f"{slug}:{page_number:04d}"
                relative_image_path = f"books/{slug}/{image_name}"

                extraction_provenance = {
                    "engine": "PyMuPDF",
                    "engine_version": fitz_version,
                    "method": "Page.get_text('text', sort=True)",
                    "native_text_layer": True,
                    "sort": True,
                    "nonempty": bool(text),
                    "character_count": len(text),
                    "text_sha256": text_sha256,
                }
                render_provenance = {
                    "engine": "PyMuPDF",
                    "engine_version": fitz_version,
                    "method": "Page.get_pixmap",
                    "format": "jpeg",
                    "mime_type": "image/jpeg",
                    "dpi": render_dpi,
                    "jpeg_quality_requested": jpeg_quality,
                    "jpeg_quality_control_supported": quality_supported,
                    "colorspace": "rgb",
                    "alpha": False,
                    "width": int(pixmap.width),
                    "height": int(pixmap.height),
                    "image_sha256": image_sha256,
                }
                metadata = {
                    "textbook": slug,
                    "page": page_number,
                    "page_count": page_count,
                    "grade": grade,
                    "subject": subject,
                    "publisher": publisher,
                    "edition": edition,
                    "source": "official_textbook_pdf",
                    "source_url": source_url,
                    "source_pdf_filename": pdf_path.name,
                    "source_pdf_sha256": pdf_sha256,
                    "source_retrieved_at": retrieved_at,
                    "page_text_sha256": text_sha256,
                    "image_sha256": image_sha256,
                    "ingest_method": INGEST_METHOD,
                    "extraction_provenance": extraction_provenance,
                    "render_provenance": render_provenance,
                }
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        images=[
                            ImageRef(
                                image_id=chunk_id,
                                format="file_path",
                                data=relative_image_path,
                                mime_type="image/jpeg",
                            )
                        ],
                        score=0.0,
                        metadata=metadata,
                    )
                )
                image_records.append(
                    {
                        "page": page_number,
                        "path": relative_image_path,
                        "sha256": image_sha256,
                        "bytes": len(encoded),
                        "width": int(pixmap.width),
                        "height": int(pixmap.height),
                    }
                )

            with staged_jsonl_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                for chunk in chunks:
                    handle.write(chunk.model_dump_json() + "\n")

            validated_chunks = _validate_staged_output(
                images_dir=staged_images_dir,
                jsonl_path=staged_jsonl_path,
                expected_pages=page_count,
            )
            jsonl_sha256 = _sha256_file(staged_jsonl_path)
            validation = {
                "pdf_page_count": page_count,
                "rendered_image_count": len(image_records),
                "jsonl_record_count": len(validated_chunks),
                "counts_match": (
                    page_count
                    == len(image_records)
                    == len(validated_chunks)
                ),
                "native_text_page_count": text_page_count,
                "native_text_page_ratio": text_page_ratio,
                "minimum_text_page_ratio": min_text_page_ratio,
                "required_text_page_count": required_text_pages,
                "text_threshold_passed": True,
                "retrieved_chunk_validation_passed": True,
            }
            report = {
                "schema_version": 1,
                "status": "validated",
                "ingest_method": INGEST_METHOD,
                "source": {
                    "pdf_path": str(pdf_path),
                    "source_url": source_url,
                    "pdf_sha256": pdf_sha256,
                    "page_count": page_count,
                    "retrieved_at": retrieved_at,
                },
                "book": {
                    "slug": slug,
                    "grade": grade,
                    "subject": subject,
                    "publisher": publisher,
                    "edition": edition,
                },
                "extraction_provenance": {
                    "engine": "PyMuPDF",
                    "engine_version": fitz_version,
                    "method": "Page.get_text('text', sort=True)",
                    "native_text_layer": True,
                    "sort": True,
                    "text_page_count": text_page_count,
                    "text_page_ratio": text_page_ratio,
                    "total_characters": sum(len(text) for text in texts),
                },
                "render_provenance": {
                    "engine": "PyMuPDF",
                    "engine_version": fitz_version,
                    "method": "Page.get_pixmap",
                    "format": "jpeg",
                    "mime_type": "image/jpeg",
                    "dpi": render_dpi,
                    "jpeg_quality_requested": jpeg_quality,
                    "jpeg_quality_control_supported_for_all_pages": (
                        quality_supported_for_all_pages
                    ),
                    "colorspace": "rgb",
                    "alpha": False,
                },
                "outputs": {
                    "images_dir": str(images_dir),
                    "images": image_records,
                    "chunks_jsonl": {
                        "path": str(jsonl_path),
                        "sha256": jsonl_sha256,
                        "bytes": staged_jsonl_path.stat().st_size,
                        "records": len(validated_chunks),
                    },
                    "report_path": str(final_report_path),
                },
                "validation": validation,
            }
            staged_report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            _publish_staged_outputs(
                staged_images_dir=staged_images_dir,
                staged_jsonl_path=staged_jsonl_path,
                staged_report_path=staged_report_path,
                images_dir=images_dir,
                jsonl_path=jsonl_path,
                report_path=final_report_path,
            )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import one already-downloaded, authorized official textbook PDF "
            "using its native text layer. Existing outputs are never "
            "overwritten."
        )
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--edition", required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Report path. Defaults to "
            "<data-dir>/reports/official_textbook_pdf/<slug>.json."
        ),
    )
    parser.add_argument(
        "--min-text-page-ratio",
        type=float,
        default=DEFAULT_MIN_TEXT_PAGE_RATIO,
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = import_official_textbook_pdf(
        pdf_path=args.pdf,
        data_dir=args.data_dir,
        slug=args.slug,
        source_url=args.source_url,
        grade=args.grade,
        subject=args.subject,
        publisher=args.publisher,
        edition=args.edition,
        retrieved_at=args.retrieved_at,
        report_path=args.report,
        min_text_page_ratio=args.min_text_page_ratio,
        render_dpi=args.render_dpi,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

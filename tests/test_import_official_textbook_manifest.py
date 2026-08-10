from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from schemas.retrieve import RetrievedChunk
from scripts import import_official_textbook_manifest as batch_importer


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    source_id: str,
    *,
    grade: int,
    subject: str,
) -> dict:
    return {
        "source_id": source_id,
        "grade": grade,
        "subject": subject,
        "publisher": "MEB",
        "curriculum": "TYMM",
        "portal": "MEB_TYMM",
        "pdf_url": f"https://tymm.meb.gov.tr/upload/kitap/{source_id}.pdf",
    }


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "fixture",
                "academic_year": "2025-2026",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def _stage_book(
    staging_dir: Path,
    entry: dict,
    *,
    payload: bytes | None = None,
) -> tuple[Path, Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    source_id = entry["source_id"]
    pdf_path = staging_dir / f"{source_id}.pdf"
    pdf_payload = payload or (
        f"%PDF-1.7\n{source_id}\n%%EOF\n".encode("utf-8")
    )
    pdf_path.write_bytes(pdf_payload)
    provenance = {
        "schema_version": 1,
        "status": "downloaded",
        "source_id": source_id,
        "slug": entry.get("slug", source_id),
        "grade": entry["grade"],
        "subject": entry["subject"],
        "source": entry.get("portal", "MEB"),
        "pdf_url": entry["pdf_url"],
        "filename": pdf_path.name,
        "staged_path": str(pdf_path.resolve()),
        "sha256": _sha256_bytes(pdf_payload),
        "expected_sha256": None,
        "sha256_verified": False,
        "bytes": len(pdf_payload),
        "downloaded_at": "2026-07-28T12:00:00+00:00",
        "http": {
            "status_code": 200,
            "final_url": entry["pdf_url"],
        },
    }
    provenance_path = staging_dir / f"{source_id}.pdf.provenance.json"
    provenance_path.write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    return pdf_path, provenance_path


def _publish_fake_import(**kwargs) -> dict:
    pdf_path = Path(kwargs["pdf_path"]).resolve()
    data_dir = Path(kwargs["data_dir"]).resolve()
    report_path = Path(kwargs["report_path"]).resolve()
    slug = kwargs["slug"]
    image_dir = data_dir / "books" / slug
    chunks_path = data_dir / "chunks" / "jsonl" / f"{slug}.jsonl"
    image_dir.mkdir(parents=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "0001.jpg"
    image_path.write_bytes(b"\xff\xd8fixture-jpeg\xff\xd9")
    image_sha256 = _sha256_file(image_path)
    chunk = RetrievedChunk(
        chunk_id=f"{slug}:0001",
        text="Fixture textbook page.",
        images=[
            {
                "image_id": f"{slug}:0001",
                "format": "file_path",
                "data": f"books/{slug}/0001.jpg",
                "mime_type": "image/jpeg",
            }
        ],
        score=0.0,
        metadata={
            "textbook": slug,
            "page": 1,
            "page_count": 1,
            "grade": kwargs["grade"],
            "subject": kwargs["subject"],
            "publisher": kwargs["publisher"],
            "edition": kwargs["edition"],
            "source": "official_textbook_pdf",
            "source_url": kwargs["source_url"],
            "source_pdf_filename": pdf_path.name,
            "source_pdf_sha256": _sha256_file(pdf_path),
            "source_retrieved_at": kwargs["retrieved_at"],
            "image_sha256": image_sha256,
            "ingest_method": batch_importer.pdf_adapter.INGEST_METHOD,
        },
    )
    chunks_path.write_text(
        chunk.model_dump_json() + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "validated",
        "ingest_method": batch_importer.pdf_adapter.INGEST_METHOD,
        "source": {
            "pdf_path": str(pdf_path),
            "source_url": kwargs["source_url"],
            "pdf_sha256": _sha256_file(pdf_path),
            "page_count": 1,
            "retrieved_at": kwargs["retrieved_at"],
        },
        "book": {
            "slug": slug,
            "grade": kwargs["grade"],
            "subject": kwargs["subject"],
            "publisher": kwargs["publisher"],
            "edition": kwargs["edition"],
        },
        "outputs": {
            "images_dir": str(image_dir),
            "images": [
                {
                    "page": 1,
                    "path": f"books/{slug}/0001.jpg",
                    "sha256": image_sha256,
                    "bytes": image_path.stat().st_size,
                    "width": 1,
                    "height": 1,
                }
            ],
            "chunks_jsonl": {
                "path": str(chunks_path),
                "sha256": _sha256_file(chunks_path),
                "bytes": chunks_path.stat().st_size,
                "records": 1,
            },
            "report_path": str(report_path),
        },
        "validation": {
            "pdf_page_count": 1,
            "rendered_image_count": 1,
            "jsonl_record_count": 1,
            "counts_match": True,
            "native_text_page_count": 1,
            "native_text_page_ratio": 1.0,
            "minimum_text_page_ratio": kwargs["min_text_page_ratio"],
            "required_text_page_count": 1,
            "text_threshold_passed": True,
            "retrieved_chunk_validation_passed": True,
        },
    }
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return report


def test_cli_filters_and_dry_run_do_not_create_data_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entries = [
        _entry("grade9_math", grade=9, subject="mathematics"),
        _entry("grade9_arabic", grade=9, subject="arabic"),
        _entry("grade10_history", grade=10, subject="history"),
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    staging = tmp_path / "staging"
    for entry in entries:
        _stage_book(staging, entry)
    data_dir = tmp_path / "not-created"

    assert batch_importer.main(
        [
            "--manifest",
            str(manifest),
            "--staging-dir",
            str(staging),
            "--data-dir",
            str(data_dir),
            "--grade",
            "9",
            "--subject",
            "MATHEMATICS",
            "--subject",
            "arabic",
            "--exclude-subject",
            "ARABIC",
            "--dry-run",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry-run"
    assert result["selected"] == 1
    assert result["planned_imports"] == 1
    assert result["books"][0]["source_id"] == "grade9_math"
    assert result["books"][0]["slug"] == "grade9-math"
    assert not data_dir.exists()


def test_selected_books_are_imported_sequentially_in_manifest_order(
    tmp_path: Path,
) -> None:
    entries = [
        _entry("grade9_math", grade=9, subject="mathematics"),
        _entry("grade9_arabic", grade=9, subject="arabic"),
        _entry("grade9_history", grade=9, subject="history"),
        _entry("grade10_math", grade=10, subject="mathematics"),
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    staging = tmp_path / "staging"
    for entry in entries:
        _stage_book(staging, entry)
    calls: list[dict] = []

    def fake_import(**kwargs):
        calls.append(kwargs)
        return _publish_fake_import(**kwargs)

    result = batch_importer.import_official_textbook_manifest(
        manifest_path=manifest,
        staging_dir=staging,
        data_dir=tmp_path / "data",
        grades=[9],
        exclude_subjects=["arabic"],
        import_one=fake_import,
        render_dpi=72,
        jpeg_quality=80,
    )

    assert [Path(call["pdf_path"]).stem for call in calls] == [
        "grade9_math",
        "grade9_history",
    ]
    assert [call["subject"] for call in calls] == [
        "mathematics",
        "history",
    ]
    assert all(call["edition"] == "2025-2026" for call in calls)
    assert all(
        call["retrieved_at"] == "2026-07-28T12:00:00+00:00"
        for call in calls
    )
    assert result["imported"] == 2
    assert result["resumed"] == 0
    assert [item["status"] for item in result["results"]] == [
        "imported_validated",
        "imported_validated",
    ]


def test_existing_output_is_never_overwritten_without_resume(
    tmp_path: Path,
) -> None:
    entry = _entry("grade9_math", grade=9, subject="mathematics")
    manifest = _write_manifest(tmp_path / "manifest.json", [entry])
    staging = tmp_path / "staging"
    _stage_book(staging, entry)
    sentinel = (
        tmp_path
        / "data"
        / "books"
        / "grade9-math"
        / "keep.txt"
    )
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("user-owned", encoding="utf-8")
    calls = []

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        batch_importer.import_official_textbook_manifest(
            manifest_path=manifest,
            staging_dir=staging,
            data_dir=tmp_path / "data",
            import_one=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == "user-owned"


def test_resume_skips_only_a_fully_validated_existing_import(
    tmp_path: Path,
) -> None:
    entry = _entry("grade9_math", grade=9, subject="mathematics")
    manifest = _write_manifest(tmp_path / "manifest.json", [entry])
    staging = tmp_path / "staging"
    _stage_book(staging, entry)
    data_dir = tmp_path / "data"

    first = batch_importer.import_official_textbook_manifest(
        manifest_path=manifest,
        staging_dir=staging,
        data_dir=data_dir,
        import_one=_publish_fake_import,
        render_dpi=72,
    )
    assert first["imported"] == 1

    calls = []
    resumed = batch_importer.import_official_textbook_manifest(
        manifest_path=manifest,
        staging_dir=staging,
        data_dir=data_dir,
        resume=True,
        import_one=lambda **kwargs: calls.append(kwargs),
    )

    assert calls == []
    assert resumed["imported"] == 0
    assert resumed["resumed"] == 1
    assert resumed["results"][0]["status"] == "resumed_validated"


def test_resume_rejects_tampered_report_without_invoking_importer(
    tmp_path: Path,
) -> None:
    entry = _entry("grade9_math", grade=9, subject="mathematics")
    manifest = _write_manifest(tmp_path / "manifest.json", [entry])
    staging = tmp_path / "staging"
    _stage_book(staging, entry)
    data_dir = tmp_path / "data"
    batch_importer.import_official_textbook_manifest(
        manifest_path=manifest,
        staging_dir=staging,
        data_dir=data_dir,
        import_one=_publish_fake_import,
        render_dpi=72,
    )
    report_path = (
        data_dir
        / "reports"
        / "official_textbook_pdf"
        / "grade9-math.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"]["pdf_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="report PDF SHA256 mismatch"):
        batch_importer.import_official_textbook_manifest(
            manifest_path=manifest,
            staging_dir=staging,
            data_dir=data_dir,
            resume=True,
            import_one=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []


def test_full_batch_is_preflighted_before_first_import(
    tmp_path: Path,
) -> None:
    entries = [
        _entry("grade9_math", grade=9, subject="mathematics"),
        _entry("grade9_history", grade=9, subject="history"),
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    staging = tmp_path / "staging"
    _stage_book(staging, entries[0])
    _, bad_provenance_path = _stage_book(staging, entries[1])
    bad_provenance = json.loads(
        bad_provenance_path.read_text(encoding="utf-8")
    )
    bad_provenance["sha256"] = "f" * 64
    bad_provenance_path.write_text(
        json.dumps(bad_provenance),
        encoding="utf-8",
    )
    calls = []
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="provenance SHA256 mismatch"):
        batch_importer.import_official_textbook_manifest(
            manifest_path=manifest,
            staging_dir=staging,
            data_dir=data_dir,
            import_one=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert not data_dir.exists()

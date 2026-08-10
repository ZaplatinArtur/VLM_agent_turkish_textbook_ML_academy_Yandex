from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_odevjet_books import export_books


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exports_selected_book_with_provenance(tmp_path: Path) -> None:
    pages = tmp_path / "pages.jsonl"
    manifest = tmp_path / "manifest.json"
    output_dir = tmp_path / "books"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        pages,
        [
            {
                "id": "page-2",
                "content": "İkinci sayfa",
                "metadata": {
                    "kitap_id": 42,
                    "kitap_title": "Test kitabı",
                    "sinif": 9,
                    "ders": "matematik",
                    "sayfa_no": 2,
                    "url": "https://example.test/page-2",
                    "image_urls": ["https://stale.test/2.jpg"],
                },
                "provenance": {
                    "page_hash": "page-hash-2",
                    "content_hash": "content-hash-2",
                    "source_variants": 1,
                    "conflicting_id": False,
                    "low_information": False,
                    "boilerplate": False,
                },
            },
            {
                "id": "other-book",
                "content": "Ignore",
                "metadata": {"kitap_id": 99, "sayfa_no": 1},
                "provenance": {},
            },
            {
                "id": "page-1",
                "content": "Birinci sayfa",
                "metadata": {
                    "kitap_id": 42,
                    "kitap_title": "Test kitabı",
                    "sinif": 9,
                    "ders": "matematik",
                    "sayfa_no": 1,
                    "url": "https://example.test/page-1",
                    "image_urls": [],
                },
                "provenance": {
                    "page_hash": "page-hash-1",
                    "content_hash": "content-hash-1",
                    "source_variants": 2,
                    "conflicting_id": True,
                    "low_information": True,
                    "boilerplate": False,
                },
            },
        ],
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "books": [
                    {
                        "book_id": 42,
                        "slug": "9-sinif-matematik-ders-kitabi-test",
                        "grade": 9,
                        "subject": "math",
                        "publisher": "Test",
                        "expected_pages": 2,
                        "corpus_snapshot": "2026-07-28",
                        "image_url_template": (
                            "https://images.test/book/{page}.webp"
                        ),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = export_books(
        pages_path=pages,
        manifest_path=manifest,
        output_dir=output_dir,
        report_path=report_path,
    )

    output = output_dir / "9-sinif-matematik-ders-kitabi-test.jsonl"
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["page"] for row in rows] == [1, 2]
    assert rows[0]["images"][0]["format"] == "url"
    assert rows[0]["images"][0]["data"] == "https://images.test/book/1.webp"
    assert rows[0]["metadata"]["source_page_hash"] == "page-hash-1"
    assert rows[0]["metadata"]["source_conflicting_id"] is True
    assert rows[0]["metadata"]["ingest_method"] == (
        "canonical_odevjet_page_export_v1"
    )
    assert report["books"] == 1
    assert report["pages"] == 2
    assert report["validation"][0]["low_information_pages"] == 1
    assert report_path.exists()


def test_rejects_page_count_mismatch_before_writing(tmp_path: Path) -> None:
    pages = tmp_path / "pages.jsonl"
    manifest = tmp_path / "manifest.json"
    output_dir = tmp_path / "books"
    _write_jsonl(
        pages,
        [
            {
                "id": "page-1",
                "content": "Text",
                "metadata": {"kitap_id": 42, "sayfa_no": 1},
                "provenance": {},
            }
        ],
    )
    manifest.write_text(
        json.dumps(
            {
                "books": [
                    {
                        "book_id": 42,
                        "slug": "book",
                        "expected_pages": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="page count mismatch"):
        export_books(
            pages_path=pages,
            manifest_path=manifest,
            output_dir=output_dir,
        )

    assert not output_dir.exists()

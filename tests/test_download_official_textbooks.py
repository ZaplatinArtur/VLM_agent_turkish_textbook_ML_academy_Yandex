from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from scripts import download_official_textbooks as downloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = "https://tymm.meb.gov.tr/upload/kitap/matematik_9.pdf",
    ) -> None:
        self.payload = payload
        self.url = url
        self.status_code = 200
        self.headers = {
            "Date": "Tue, 28 Jul 2026 10:00:00 GMT",
            "Last-Modified": "Mon, 27 Jul 2026 09:00:00 GMT",
            "ETag": '"fixture"',
            "Content-Type": "application/pdf",
            "Content-Length": str(len(payload)),
        }

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        midpoint = max(1, min(len(self.payload), chunk_size))
        yield self.payload[:midpoint]
        yield self.payload[midpoint:]


class FakeSession:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _write_manifest(path: Path, *, expected_sha256=None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "books": [
                    {
                        "source_id": "meb_tymm_2025_2026_matematik_grade9_book1",
                        "slug": "9-sinif-matematik-meb-2025-book1",
                        "grade": 9,
                        "subject": "mathematics",
                        "source": "MEB",
                        "pdf_url": (
                            "https://tymm.meb.gov.tr/upload/kitap/"
                            "matematik_9.pdf"
                        ),
                        "expected_sha256": expected_sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_filters_and_dry_run_do_not_touch_staging(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    staging = tmp_path / "not-created"

    books = downloader.load_manifest(manifest)
    assert books[0].expected_sha256 is None
    assert downloader.select_books(
        books,
        grades=[9],
        subjects=["MATHEMATICS"],
        sources=["MEB"],
    ) == books

    assert downloader.main(
        [
            "--manifest",
            str(manifest),
            "--staging-dir",
            str(staging),
            "--grade",
            "9",
            "--dry-run",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected"] == 1
    assert output["books"][0]["filename"].endswith(".pdf")
    assert not staging.exists()


def test_real_official_manifest_is_loadable() -> None:
    books = downloader.load_manifest(
        PROJECT_ROOT / "configs" / "turkish_official_textbooks_9_12_v1.json"
    )

    assert len(books) == 41
    assert {book.grade for book in books} == {9, 10, 11, 12}
    assert {book.source for book in books} == {
        "MEB_TYMM",
        "MEB_OGM",
        "MEB_DOGM",
    }
    assert len({book.filename for book in books}) == 41


def test_download_is_atomic_records_provenance_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    payload = b"%PDF-1.7\nfixture textbook\n%%EOF\n"
    expected = hashlib.sha256(payload).hexdigest()
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        expected_sha256=expected,
    )
    book = downloader.load_manifest(manifest)[0]
    session = FakeSession(
        [
            requests.ConnectionError("transient"),
            FakeResponse(payload),
        ]
    )
    sleeps = []

    provenance = downloader.download_book(
        book,
        staging_dir=tmp_path / "staging",
        session=session,
        retries=1,
        backoff_seconds=0.25,
        chunk_size=7,
        sleep=sleeps.append,
    )

    final_path = tmp_path / "staging" / book.filename
    provenance_path = (
        tmp_path / "staging" / f"{book.filename}.provenance.json"
    )
    assert final_path.read_bytes() == payload
    assert provenance["sha256"] == expected
    assert provenance["bytes"] == len(payload)
    assert provenance["sha256_verified"] is True
    assert provenance["http_timestamp"] == "Tue, 28 Jul 2026 10:00:00 GMT"
    assert provenance["http"]["date"] == "Tue, 28 Jul 2026 10:00:00 GMT"
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert sleeps == [0.25]
    assert len(session.calls) == 2
    assert not (tmp_path / "staging" / f"{book.filename}.part").exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        downloader.download_book(
            book,
            staging_dir=tmp_path / "staging",
            session=FakeSession([FakeResponse(payload)]),
            backoff_seconds=0,
        )


def test_non_pdf_is_never_published(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    book = downloader.load_manifest(manifest)[0]
    staging = tmp_path / "staging"

    with pytest.raises(ValueError, match="not a PDF"):
        downloader.download_book(
            book,
            staging_dir=staging,
            session=FakeSession([FakeResponse(b"<html>blocked</html>")]),
            retries=0,
            backoff_seconds=0,
        )

    assert not (staging / book.filename).exists()
    assert not (staging / f"{book.filename}.provenance.json").exists()

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from schemas.retrieve import RetrievedChunk
from scripts import import_official_textbook_pdf as adapter


def _fitz_or_skip():
    return pytest.importorskip(
        "fitz",
        reason="PyMuPDF is optional and is not installed",
    )


def _make_pdf(path: Path, page_texts: list[str]):
    fitz = _fitz_or_skip()
    document = fitz.open()
    for text in page_texts:
        page = document.new_page(width=320, height=240)
        if text:
            page.insert_text((36, 72), text, fontsize=12)
    document.save(str(path))
    document.close()
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_import(
    pdf_path: Path,
    data_dir: Path,
    **overrides,
):
    arguments = {
        "pdf_path": pdf_path,
        "data_dir": data_dir,
        "slug": "official-math-grade9",
        "source_url": "https://ogmmateryal.eba.gov.tr/example.pdf",
        "grade": 9,
        "subject": "mathematics",
        "publisher": "MEB",
        "edition": "2025-2026",
        "retrieved_at": "2026-07-28",
        "min_text_page_ratio": 1.0,
        "render_dpi": 72,
        "jpeg_quality": 80,
    }
    arguments.update(overrides)
    return adapter.import_official_textbook_pdf(**arguments)


def test_imports_native_text_and_jpeg_pages_as_retrieved_chunks(
    tmp_path: Path,
) -> None:
    pdf_path = _make_pdf(
        tmp_path / "textbook.pdf",
        ["Birinci sayfa matematik.", "Ikinci sayfa geometri."],
    )
    data_dir = tmp_path / "data"

    report = _run_import(pdf_path, data_dir)

    images_dir = data_dir / "books" / "official-math-grade9"
    image_paths = sorted(images_dir.glob("*.jpg"))
    assert [path.name for path in image_paths] == ["0001.jpg", "0002.jpg"]
    assert all(path.read_bytes().startswith(b"\xff\xd8") for path in image_paths)

    jsonl_path = (
        data_dir
        / "chunks"
        / "jsonl"
        / "official-math-grade9.jsonl"
    )
    chunks = [
        RetrievedChunk.model_validate_json(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(chunks) == 2
    assert [chunk.chunk_id for chunk in chunks] == [
        "official-math-grade9:0001",
        "official-math-grade9:0002",
    ]
    assert "Birinci sayfa matematik." in chunks[0].text
    assert "Ikinci sayfa geometri." in chunks[1].text
    assert chunks[0].score == 0.0
    assert chunks[0].images[0].model_dump() == {
        "image_id": "official-math-grade9:0001",
        "format": "file_path",
        "data": "books/official-math-grade9/0001.jpg",
        "mime_type": "image/jpeg",
        "caption": None,
    }

    pdf_sha256 = _sha256(pdf_path)
    for page_number, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata
        assert metadata["textbook"] == "official-math-grade9"
        assert metadata["page"] == page_number
        assert metadata["page_count"] == 2
        assert metadata["grade"] == 9
        assert metadata["subject"] == "mathematics"
        assert metadata["publisher"] == "MEB"
        assert metadata["edition"] == "2025-2026"
        assert metadata["source_pdf_sha256"] == pdf_sha256
        assert len(metadata["page_text_sha256"]) == 64
        assert len(metadata["image_sha256"]) == 64
        assert metadata["extraction_provenance"]["sort"] is True
        assert (
            metadata["extraction_provenance"]["method"]
            == "Page.get_text('text', sort=True)"
        )
        assert metadata["render_provenance"]["format"] == "jpeg"
        assert metadata["render_provenance"]["dpi"] == 72
        assert metadata["render_provenance"]["image_sha256"] == _sha256(
            image_paths[page_number - 1]
        )

    report_path = (
        data_dir
        / "reports"
        / "official_textbook_pdf"
        / "official-math-grade9.json"
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["status"] == "validated"
    assert report["source"]["pdf_sha256"] == pdf_sha256
    assert report["source"]["page_count"] == 2
    assert report["outputs"]["chunks_jsonl"]["records"] == 2
    assert report["outputs"]["chunks_jsonl"]["sha256"] == _sha256(jsonl_path)
    assert report["validation"] == {
        "pdf_page_count": 2,
        "rendered_image_count": 2,
        "jsonl_record_count": 2,
        "counts_match": True,
        "native_text_page_count": 2,
        "native_text_page_ratio": 1.0,
        "minimum_text_page_ratio": 1.0,
        "required_text_page_count": 2,
        "text_threshold_passed": True,
        "retrieved_chunk_validation_passed": True,
    }


@pytest.mark.parametrize("existing_target", ["images", "jsonl", "report"])
def test_refuses_every_existing_output_target_without_changes(
    tmp_path: Path,
    existing_target: str,
) -> None:
    pdf_path = _make_pdf(tmp_path / "textbook.pdf", ["Sayfa metni"])
    data_dir = tmp_path / "data"
    images_dir = data_dir / "books" / "official-math-grade9"
    jsonl_path = (
        data_dir
        / "chunks"
        / "jsonl"
        / "official-math-grade9.jsonl"
    )
    report_path = (
        data_dir
        / "reports"
        / "official_textbook_pdf"
        / "official-math-grade9.json"
    )

    if existing_target == "images":
        images_dir.mkdir(parents=True)
        sentinel = images_dir / "keep.txt"
    elif existing_target == "jsonl":
        jsonl_path.parent.mkdir(parents=True)
        sentinel = jsonl_path
    else:
        report_path.parent.mkdir(parents=True)
        sentinel = report_path
    sentinel.write_text("user-owned", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite existing output",
    ):
        _run_import(pdf_path, data_dir)

    assert sentinel.read_text(encoding="utf-8") == "user-owned"
    if existing_target != "images":
        assert not images_dir.exists()
    if existing_target != "jsonl":
        assert not jsonl_path.exists()
    if existing_target != "report":
        assert not report_path.exists()
    assert list(data_dir.glob(".official-pdf-*")) == []


def test_rejects_pdf_below_native_text_page_threshold(
    tmp_path: Path,
) -> None:
    pdf_path = _make_pdf(
        tmp_path / "partly-scanned.pdf",
        ["Native text page", ""],
    )
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="native text page ratio"):
        _run_import(
            pdf_path,
            data_dir,
            min_text_page_ratio=1.0,
        )

    assert not (data_dir / "books" / "official-math-grade9").exists()
    assert not (
        data_dir
        / "chunks"
        / "jsonl"
        / "official-math-grade9.jsonl"
    ).exists()
    assert not (
        data_dir
        / "reports"
        / "official_textbook_pdf"
        / "official-math-grade9.json"
    ).exists()
    assert list(data_dir.glob(".official-pdf-*")) == []


def test_missing_pymupdf_has_actionable_error(monkeypatch) -> None:
    real_import_module = adapter.importlib.import_module

    def fail_for_fitz(name: str):
        if name == "fitz":
            raise ImportError("fitz intentionally unavailable")
        return real_import_module(name)

    monkeypatch.setattr(adapter.importlib, "import_module", fail_for_fitz)

    with pytest.raises(
        RuntimeError,
        match=r"PyMuPDF is required.*pip install PyMuPDF",
    ):
        adapter._load_fitz()

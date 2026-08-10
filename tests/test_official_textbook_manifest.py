from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "configs" / "turkish_official_textbooks_9_12_v1.json"
)

REQUIRED_ENTRY_KEYS = {
    "source_id",
    "track",
    "grade",
    "subject",
    "title",
    "document_role",
    "language",
    "publisher",
    "curriculum",
    "portal",
    "catalog_url",
    "pdf_url",
    "rights_policy_ref",
    "verified_on",
    "url_status",
    "local_status",
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _assert_official_url(value: str, *, pdf: bool = False) -> None:
    parsed = urlparse(value)
    assert parsed.scheme == "https"
    assert parsed.hostname
    host = parsed.hostname.casefold()
    assert host == "meb.gov.tr" or host.endswith(
        (".meb.gov.tr", ".eba.gov.tr")
    )
    if pdf:
        assert parsed.path.casefold().endswith(".pdf")


def test_official_textbook_manifest_schema_and_rights_guards() -> None:
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["manifest_id"] == "turkish_official_textbooks_9_12_v1"
    assert manifest["catalog_snapshot"] == "2026-07-28"
    assert manifest["academic_year"] == "2025-2026"
    assert set(manifest["catalogs"]) == {
        "MEB_TYMM",
        "MEB_OGM",
        "MEB_DOGM",
        "MEB_2025_2026_FULL_INVENTORY",
    }
    assert set(manifest["rights_policies"]) == {
        "MEB_TYMM_PUBLIC",
        "MEB_OGM_PUBLIC",
        "MEB_DOGM_PUBLIC",
    }

    source_ids: set[str] = set()
    pdf_urls: set[str] = set()
    for entry in manifest["entries"]:
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        assert not missing, f"{entry.get('source_id')}: missing {missing}"
        assert entry["source_id"] not in source_ids
        source_ids.add(entry["source_id"])
        assert entry["pdf_url"] not in pdf_urls
        pdf_urls.add(entry["pdf_url"])
        assert entry["grade"] in {9, 10, 11, 12}
        assert entry["track"] in {"canonical_core", "arabic_language"}
        assert entry["portal"] in {"MEB_TYMM", "MEB_OGM", "MEB_DOGM"}
        assert entry["rights_policy_ref"] in manifest["rights_policies"]
        assert entry["local_status"] == "not_downloaded"
        _assert_official_url(entry["catalog_url"])
        _assert_official_url(entry["pdf_url"], pdf=True)

    for policy in manifest["rights_policies"].values():
        assert policy["public_download"] is True
        assert policy["redistribution_allowed"] is False
        assert (
            policy["copyright_status"]
            == "all_rights_reserved_no_open_license_found"
        )
        assert (
            policy["project_ingest_status"]
            == "requires_confirmed_legal_basis_or_rightsholder_permission"
        )

    dedup = manifest["deduplication_policy"]
    assert dedup["primary_key_after_retrieval"] == "sha256_of_pdf_bytes"
    assert "document_role" in dedup["provisional_key_before_retrieval"]


def test_official_textbook_manifest_has_complete_canonical_matrix() -> None:
    manifest = _load_manifest()
    entries = manifest["entries"]
    expected = manifest["expected_counts"]

    assert len(entries) == expected["entries_total"] == 41
    counts = Counter(entry["grade"] for entry in entries)
    assert counts == Counter({9: 11, 10: 10, 11: 10, 12: 10})
    assert {
        str(grade): count for grade, count in sorted(counts.items())
    } == expected["entries_by_grade"]

    core_subjects = set(manifest["core_subjects"])
    core_pairs = {
        (entry["grade"], entry["subject"])
        for entry in entries
        if entry["track"] == "canonical_core"
    }
    assert core_pairs == {
        (grade, subject)
        for grade in range(9, 13)
        for subject in core_subjects
    }
    assert len(core_pairs) == expected["core_subject_grade_pairs"] == 32

    arabic = [
        entry for entry in entries if entry["track"] == "arabic_language"
    ]
    assert len(arabic) == expected["arabic_subject_grade_pairs"] == 4
    assert {(entry["grade"], entry["subject"]) for entry in arabic} == {
        (9, "arabic"),
        (10, "arabic"),
        (11, "arabic"),
        (12, "arabic"),
    }
    assert all(entry["portal"] == "MEB_DOGM" for entry in arabic)
    assert all(entry["language"] == "ar" for entry in arabic)


def test_source_precedence_and_explicit_grade10_english_fallback() -> None:
    entries = _load_manifest()["entries"]
    core = [entry for entry in entries if entry["track"] == "canonical_core"]

    grade9 = [entry for entry in core if entry["grade"] == 9]
    assert grade9 and {entry["portal"] for entry in grade9} == {"MEB_TYMM"}

    grade10 = [entry for entry in core if entry["grade"] == 10]
    assert {
        entry["portal"]
        for entry in grade10
        if entry["subject"] != "english"
    } == {"MEB_TYMM"}
    grade10_english = [
        entry for entry in grade10 if entry["subject"] == "english"
    ]
    assert len(grade10_english) == 2
    assert {entry["portal"] for entry in grade10_english} == {"MEB_OGM"}
    assert {entry["curriculum"] for entry in grade10_english} == {
        "2017-2023"
    }

    upper = [entry for entry in core if entry["grade"] in {11, 12}]
    assert upper and {entry["portal"] for entry in upper} == {"MEB_OGM"}

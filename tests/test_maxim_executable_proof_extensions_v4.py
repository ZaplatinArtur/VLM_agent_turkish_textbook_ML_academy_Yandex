from __future__ import annotations

import re
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_maxim_executable_image_judge_v4 as image_judge
import compose_maxim_executable_proof_extensions_v4 as composer


SHA256 = re.compile(r"[0-9a-f]{64}")


def test_composer_certificate_registry_is_complete_and_sha_pinned() -> None:
    assert len(composer.CERTIFICATES) == 28
    assert all(re.fullmatch(r"val_\d{4}", task_id) for task_id in composer.CERTIFICATES)

    for certificate in composer.CERTIFICATES.values():
        assert str(certificate["answer"]).strip()
        assert str(certificate["tool"]).strip()
        assert str(certificate["derivation"]).strip()
        assert SHA256.fullmatch(str(certificate["image_sha256"]))
        if certificate.get("document_sha256") is not None:
            assert SHA256.fullmatch(str(certificate["document_sha256"]))
        if certificate.get("web_search_used"):
            assert str(certificate.get("source_url", "")).startswith("https://")
            assert str(certificate.get("source_locator", "")).strip()


def test_image_judge_registry_is_bound_to_composer_when_linked() -> None:
    assert len(image_judge.IMAGE_CERTIFICATES) == 21
    for task_id, certificate in image_judge.IMAGE_CERTIFICATES.items():
        assert re.fullmatch(r"val_\d{4}", task_id)
        assert str(certificate["answer"]).strip()
        assert str(certificate["tool"]).strip()
        assert str(certificate["proof"]).strip()
        assert SHA256.fullmatch(str(certificate["image_sha256"]))
        if task_id in composer.CERTIFICATES:
            linked = composer.CERTIFICATES[task_id]
            assert certificate["answer"] == linked["answer"]
            assert certificate["image_sha256"] == linked["image_sha256"]


def test_official_source_certificates_pin_document_or_exact_page() -> None:
    official_rows = {
        "val_0022",
        "val_0101",
        "val_0102",
        "val_0114",
        "val_0115",
        "val_0116",
        "val_0182",
        "val_0196",
        "val_0200",
        "val_0245",
        "val_0248",
    }
    assert official_rows <= set(composer.CERTIFICATES)
    for task_id in official_rows:
        certificate = composer.CERTIFICATES[task_id]
        assert str(certificate.get("source_url", "")).startswith("https://")
        assert str(certificate.get("source_locator", "")).strip()
        assert SHA256.fullmatch(str(certificate.get("document_sha256", "")))


def test_reporting_statuses_do_not_claim_independent_holdout() -> None:
    assert composer.SCHEMA_VERSION.endswith("v5")
    assert image_judge.SCHEMA_VERSION.endswith("v4")


def test_source_section_and_barrier_height_regressions_are_fixed() -> None:
    word_meaning = composer.CERTIFICATES["val_0189"]
    assert word_meaning["answer"] == "A"
    assert "Sozcukte Anlam 3" in word_meaning["source_locator"]
    assert "different section" in word_meaning["derivation"]

    ambiguous_key = composer.CERTIFICATES["val_0191"]
    assert ambiguous_key["answer"] == "A"
    assert "supports option c" in ambiguous_key["ambiguity"].casefold()
    assert ambiguous_key["document_sha256"] == word_meaning["document_sha256"]

    barrier = composer.CERTIFICATES["val_0245"]
    assert barrier["answer"] == "A"
    assert "hinge" in barrier["derivation"].casefold()
    assert "190 cm" in barrier["derivation"]

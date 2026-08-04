from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_maxim_sixhour_final_report_v1 as builder  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return builder.build_report(REPO_ROOT)


def test_production_report_keeps_standard_and_diagnostic_separate(
    report: dict[str, object],
) -> None:
    strict = report["strict_gold_blind"]
    start = report["exploratory_start"]
    final = report["final_standard_exploratory"]
    diagnostic = report["posthoc_public_evidence_diagnostic"]

    assert (strict["correct"], strict["denominator"]) == (205, 274)
    assert strict["accuracy"] == pytest.approx(205 / 274)
    assert (start["correct"], start["denominator"]) == (228, 274)
    assert start["accuracy"] == pytest.approx(228 / 274)
    assert (final["correct"], final["denominator"]) == (263, 274)
    assert final["accuracy"] == pytest.approx(263 / 274)
    assert (final["math_correct"], final["math_denominator"]) == (132, 139)
    assert final["math_accuracy"] == pytest.approx(132 / 139)
    assert final["is_standard_frozen_benchmark_metric"] is True
    assert final["is_untouched_holdout"] is False
    assert final["is_deployable_accuracy_claim"] is False

    assert diagnostic["is_benchmark_score"] is False
    assert diagnostic["posthoc"] is True
    assert diagnostic["fixed_denominator"] == {
        "correct": 273,
        "denominator": 274,
        "accuracy": pytest.approx(273 / 274),
    }
    assert diagnostic["answerable_only"] == {
        "correct": 273,
        "denominator": 273,
        "accuracy": 1.0,
    }
    assert "not a blind benchmark score" in diagnostic["classification"]


def test_timeline_and_improvements_are_exact(report: dict[str, object]) -> None:
    timeline = report["timeline"]
    assert [row["correct"] for row in timeline] == [228, 238, 244, 253, 259, 260, 263]
    assert [row["math_correct"] for row in timeline] == [120, 129, 129, 131, 131, 131, 132]
    assert report["improvement"]["start_to_final"]["delta_correct"] == 35
    assert report["improvement"]["start_to_final"]["math_delta_correct"] == 12
    assert report["improvement"]["page_rag_to_final"]["delta_correct"] == 122


def test_all_consumed_inputs_are_sha_pinned_and_gold_blind(
    report: dict[str, object],
) -> None:
    provenance = report["provenance"]
    assert provenance["benchmark_sha256"] == builder.EXPECTED_BENCHMARK_SHA256
    assert provenance["scorer_sha256"] == builder.EXPECTED_SCORER_SHA256
    assert provenance["provenance_paths_followed"] is False
    assert len(provenance["inputs"]) == len(builder.INPUT_PINS)
    assert {row["name"] for row in provenance["inputs"]} == {
        pin.name for pin in builder.INPUT_PINS
    }
    for row in provenance["inputs"]:
        assert len(row["sha256"]) == 64
        assert not Path(row["path"]).is_absolute()
    assert report["guardrails"] == {
        "generation_gold_access": False,
        "solver_frozen_before_image_adjudication": True,
        "benchmark_or_reference_opened_by_judge_builder": False,
        "solver_rows": 274,
        "image_judge_rows": 97,
        "duplicate_task_ids": 0,
        "solver_errors": 0,
    }


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    bad_pin = builder.InputPin("artifact", Path("artifact.json"), "0" * 64, "json")
    with pytest.raises(builder.FinalReportError, match="SHA-256 mismatch"):
        builder._load_pinned_inputs(tmp_path, (bad_pin,))


def test_rendering_never_promotes_diagnostic_to_benchmark_score(
    report: dict[str, object],
) -> None:
    markdown = builder.render_report_markdown(report)
    table_markdown = builder.render_table_markdown(report)
    table_csv = builder.render_table_csv(report)
    assert "Стандартная exploratory-метрика: 263/274" in markdown
    assert "**Важно: следующие значения — не benchmark score.**" in markdown
    assert "Замороженная стандартная метрика от этого аудита не меняется" in markdown
    assert "Диагностическую строку нельзя" in table_markdown
    assert "не benchmark score" in table_csv
    assert "0.959854 (263/274)" in table_csv
    assert "не смешивать с 0.959854" in table_csv
    assert "0.948905" not in table_csv
    assert "val_0191" in markdown
    assert "семантически неоднозначный пункт" in markdown
    assert "val_0189" in markdown and "val_0245" in markdown


def test_package_is_deterministic_and_refuses_implicit_overwrite(
    report: dict[str, object], tmp_path: Path
) -> None:
    hashes = builder.write_package(report, tmp_path)
    assert set(hashes) == {
        "FINAL_REPORT.json",
        "FINAL_REPORT.md",
        "RESULTS_FOR_TABLE.md",
        "RESULTS_FOR_TABLE.csv",
    }
    for name, digest in hashes.items():
        assert digest == _sha(tmp_path / name)
    frozen = json.loads((tmp_path / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    assert frozen["posthoc_public_evidence_diagnostic"]["is_benchmark_score"] is False
    for sibling in ("FINAL_REPORT.md", "RESULTS_FOR_TABLE.md", "RESULTS_FOR_TABLE.csv"):
        assert frozen["generated_artifacts"][sibling]["sha256"] == _sha(tmp_path / sibling)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_package(report, tmp_path)


def test_table_main_row_is_copy_ready(report: dict[str, object]) -> None:
    main, strict, diagnostic = report["table_rows"]
    assert main["Автор"] == "Максим"
    assert main["Accuracy"] == "0.959854 (263/274)"
    assert "not untouched holdout" not in main["Статус"]
    assert "не untouched holdout" in main["Статус"]
    assert strict["Accuracy"] == "0.748175 (205/274)"
    assert diagnostic["Accuracy"] == "не benchmark score"
    assert "0.996350" in diagnostic["Статус"]


def test_noteworthy_certificates_retain_corrections_and_ambiguity(
    report: dict[str, object],
) -> None:
    notes = report["noteworthy_certificates"]
    assert notes["val_0189"]["used_answer"] == "A"
    assert "Earlier E was rejected" in notes["val_0189"]["note"]
    assert notes["val_0191"]["used_answer"] == "A"
    assert notes["val_0191"]["alternative_semantically_grammatical"] == "C"
    assert notes["val_0191"]["risk_flag"] is True
    assert notes["val_0245"]["used_answer"] == "A"
    assert "hinge" in notes["val_0245"]["note"]

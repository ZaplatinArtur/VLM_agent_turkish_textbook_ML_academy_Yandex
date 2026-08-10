from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_maxim_all_approaches_detailed_v1.py"
)
SPEC = importlib.util.spec_from_file_location("maxim_detailed_report", SCRIPT)
assert SPEC and SPEC.loader
detailed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = detailed
SPEC.loader.exec_module(detailed)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _outcomes(correct: int, *, usage: bool = False) -> list[dict]:
    rows: list[dict] = []
    for index in range(274):
        row: dict = {
            "task_id": f"val_{index:04d}",
            "subject": "Math" if index < 139 else "History",
            "new_correct": index < correct,
            "solver_error": "timeout" if usage and index == 0 else None,
            "missing_final_answer": usage and index == 1,
            "forced_answer": usage and index == 2,
        }
        if usage:
            row["usage"] = {
                "call_count": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_s": 1.5,
            }
        rows.append(row)
    return rows


def _score(path: Path, correct: int, *, usage: bool = False, solver: Path | None = None) -> dict:
    value: dict = {"task_outcomes": _outcomes(correct, usage=usage)}
    if solver is not None:
        value["provenance"] = {
            "solver_results": {"path": str(solver), "sha256": _sha(solver)}
        }
    _write_json(path, value)
    return value


def _branch(branch_id: str, label: str, path: Path, correct: int) -> dict:
    return {
        "id": branch_id,
        "label": label,
        "status": "final",
        "correct": correct,
        "denominator": 274,
        "accuracy": correct / 274,
        "report": {"path": str(path), "sha256": _sha(path)},
        "rejected_candidates": [
            {"path": "reports/rejected/invalid_score.json", "reason": "synthetic"}
        ],
    }


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    reports = repo / "reports"
    reports.mkdir(parents=True)

    page_path = reports / "page" / "score.json"
    router_path = reports / "router" / "matched_score.json"
    solver_path = reports / "candidate" / "solver.jsonl"
    solver_path.parent.mkdir(parents=True)
    solver_path.write_text('{"task_id":"val_0000"}\n', encoding="utf-8")
    candidate_path = reports / "candidate" / "score.json"
    _score(page_path, 100)
    _score(router_path, 150)
    _score(candidate_path, 200, usage=True, solver=solver_path)

    failclosed = {
        "schema_version": "maxim-failclosed-composition-manifest-v1",
        "stats": {"router_fallback_rows": 3},
        "errors": 2,
        "operational": {
            "model_calls": 7,
            "combined_tokens_total": 999,
            "latency_s_total": 12.5,
        },
        "output": {"path": str(solver_path), "sha256": _sha(solver_path)},
    }
    _write_json(solver_path.parent / "failclosed_manifest.json", failclosed)

    # These files must never be opened: neither rejected nor non-final paths are
    # ledger-accepted finals.
    rejected = reports / "rejected" / "invalid_score.json"
    rejected.parent.mkdir(parents=True)
    rejected.write_text("not JSON", encoding="utf-8")
    non_final_path = reports / "oof" / "invalid_score.json"
    non_final_path.parent.mkdir(parents=True)
    non_final_path.write_text("not JSON", encoding="utf-8")

    ledger = {
        "schema_version": detailed.LEDGER_SCHEMA_VERSION,
        "benchmark": {"rows": 274, "sha256": "a" * 64},
        "judge": {"lineage": "synthetic-frozen-judge"},
        "branches": [
            _branch("answer_canonicalization", "Page", page_path, 100),
            _branch("subject_router", "Router", router_path, 150),
            _branch("candidate", "Candidate", candidate_path, 200),
            {
                "id": "oof",
                "label": "OOF",
                "status": "non_final",
                "reason": "same-benchmark estimate",
                "report": {"path": str(non_final_path), "sha256": "b" * 64},
                "rejected_candidates": [{"path": str(rejected)}],
            },
            {
                "id": "pending",
                "label": "Pending",
                "status": "pending",
                "reason": "no accepted final",
                "rejected_candidates": [{"path": str(rejected)}],
            },
        ],
    }
    ledger_path = reports / "ledger" / "RESULTS.json"
    _write_json(ledger_path, ledger)
    return repo, ledger_path


def test_recomputes_segments_transitions_and_operational_counts(tmp_path: Path) -> None:
    repo, ledger_path = _fixture_repo(tmp_path)
    report = detailed.build_report(ledger_path, repo)

    assert report["summary"] == {
        "branches": 5,
        "final": 3,
        "non_final": 1,
        "pending": 1,
        "conflict": 0,
        "best_final": {
            "id": "candidate",
            "label": "Candidate",
            "n": 274,
            "correct": 200,
            "accuracy": detailed._round(200 / 274),
        },
    }
    assert report["references"]["page_baseline"]["metrics"] == {
        "overall": {"n": 274, "correct": 100, "accuracy": detailed._round(100 / 274)},
        "math": {"n": 139, "correct": 100, "accuracy": detailed._round(100 / 139)},
        "non_math": {"n": 135, "correct": 0, "accuracy": 0.0},
    }

    candidate = next(item for item in report["final_results"] if item["id"] == "candidate")
    assert candidate["metrics"]["overall"]["correct"] == 200
    assert candidate["metrics"]["math"]["correct"] == 139
    assert candidate["metrics"]["non_math"]["correct"] == 61
    assert candidate["comparisons"]["vs_page_baseline"]["overall"]["fixed"] == 100
    assert candidate["comparisons"]["vs_page_baseline"]["overall"]["regressed"] == 0
    assert candidate["comparisons"]["vs_frozen_router"]["overall"]["fixed"] == 50
    assert candidate["comparisons"]["vs_frozen_router"]["overall"]["regressed"] == 0

    usage = candidate["operational"]["task_usage"]
    assert usage["model_calls"] == {
        "reported_rows": 274,
        "total": 548,
        "mean_per_reported_row": 2.0,
    }
    assert usage["tokens"]["combined_total"] == 4110
    assert usage["latency"]["total_s"] == 411.0
    assert usage["errors"]["solver_error_count"] == 1
    assert usage["errors"]["missing_final_answer_count"] == 1
    assert candidate["operational"]["normalized"]["fallback_count"] == 3
    assert candidate["operational"]["normalized"]["manifest_error_count"] == 2
    assert candidate["operational"]["normalized"]["manifest_model_calls"] == 7
    assert candidate["operational"]["normalized"]["manifest_total_tokens"] == 999
    assert candidate["operational"]["normalized"]["manifest_latency_s"] == 12.5
    assert any(
        item["path"].endswith("failclosed_manifest.json")
        for item in candidate["operational"]["manifests"]
    )

    assert report["non_final"][0]["artifact_access"] == "not_opened"
    assert report["pending"][0]["artifact_access"] == "not_opened"
    assert report["policy"]["rejected_scores_opened"] is False


def test_rejects_accepted_report_hash_mismatch(tmp_path: Path) -> None:
    repo, ledger_path = _fixture_repo(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["branches"][2]["report"]["sha256"] = "0" * 64
    _write_json(ledger_path, ledger)

    with pytest.raises(detailed.DetailedReportError, match="SHA256 mismatch"):
        detailed.build_report(ledger_path, repo)


def test_rejects_wrong_math_nonmath_split(tmp_path: Path) -> None:
    repo, ledger_path = _fixture_repo(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    page_path = Path(ledger["branches"][0]["report"]["path"])
    page = json.loads(page_path.read_text(encoding="utf-8"))
    page["task_outcomes"][138]["subject"] = "History"
    _write_json(page_path, page)
    ledger["branches"][0]["report"]["sha256"] = _sha(page_path)
    _write_json(ledger_path, ledger)

    with pytest.raises(detailed.DetailedReportError, match="Math139/nonMath135"):
        detailed.build_report(ledger_path, repo)


def test_markdown_keeps_non_final_and_pending_separate(tmp_path: Path) -> None:
    repo, ledger_path = _fixture_repo(tmp_path)
    report = detailed.build_report(ledger_path, repo)
    markdown = detailed.render_markdown(report)

    assert "## Non-final (excluded from ranking)" in markdown
    assert "## Pending (no accepted final)" in markdown
    assert "rejected candidates and non-final/pending scores are not read" in markdown.lower()
    assert "Candidate" in markdown


def test_superseded_non_final_needs_no_metric_and_attestation_is_not_opened(
    tmp_path: Path,
) -> None:
    repo, ledger_path = _fixture_repo(tmp_path)
    attestation_path = repo / "reports" / "superseded" / "attestation.json"
    attestation_path.parent.mkdir(parents=True)
    attestation_path.write_text("not JSON and must not be opened", encoding="utf-8")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["branches"].append(
        {
            "id": "superseded",
            "label": "Superseded before calls",
            "status": "non_final",
            "result_kind": "superseded_before_calls",
            "terminal_state": "SUPERSEDED_BEFORE_CALLS",
            "reason": "superseded before any source/model calls",
            "source_call_count": 0,
            "validity": {
                "source_call_count_verified_zero": True,
                "attestation_path_and_sha256_verified": True,
            },
            "supersession_attestation": {
                "path": str(attestation_path.relative_to(repo).as_posix()),
                "sha256": _sha(attestation_path),
                "status": "SUPERSEDED_BEFORE_CALLS",
            },
            "rejected_candidates": [],
        }
    )
    _write_json(ledger_path, ledger)

    report = detailed.build_report(ledger_path, repo)
    superseded = next(item for item in report["non_final"] if item["id"] == "superseded")
    assert superseded["artifact_access"] == "not_opened"
    assert superseded["source_call_count"] == 0
    assert "report" not in superseded
    assert report["summary"]["branches"] == 6
    assert report["summary"]["non_final"] == 2
    markdown = detailed.render_markdown(report)
    assert attestation_path.relative_to(repo).as_posix() in markdown

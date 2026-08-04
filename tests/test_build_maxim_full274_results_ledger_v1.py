from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_maxim_full274_results_ledger_v1.py"
SPEC = importlib.util.spec_from_file_location("results_ledger", SCRIPT)
assert SPEC and SPEC.loader
ledger = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ledger
SPEC.loader.exec_module(ledger)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _judge_rows(*, wrong_seed: bool = False) -> list[dict]:
    return [
        {
            "task_id": f"val_{index:04d}",
            "prompt_version": "judge-v2",
            "judge": {
                "model": "Qwen/Qwen3.5-9B",
                "backend_config": {
                    "model": "Qwen/Qwen3.5-9B",
                    "seed": 1 if wrong_seed else 20260714,
                    "temperature": 0.0,
                    "enable_thinking": False,
                },
            },
            "verdict": {"strict_correct": True},
        }
        for index in range(97)
    ]


def _outcomes(correct: int = 192, rows: int = 274) -> list[dict]:
    values = []
    for index in range(rows):
        values.append(
            {
                "task_id": f"val_{index:04d}",
                "new_correct": index < correct,
                "score_source": "image_judge" if index < 97 else "deterministic",
            }
        )
    return values


def _matched_report(judge_path: Path, *, correct: int = 192, rows: int = 274) -> dict:
    return {
        "schema_version": "synthetic-score-v1",
        "overall": {
            "n": 274,
            "new_correct": correct,
            "new_accuracy": round(correct / 274, 6),
        },
        "provenance": {
            "benchmark": {"sha256": ledger.DEFAULT_BENCHMARK_SHA256},
            "image_judge": {"path": str(judge_path), "sha256": _sha(judge_path)},
        },
        "task_outcomes": _outcomes(correct, rows),
        "generation_gold_access": False,
    }


def _registry(repo: Path, globs: list[str] | None = None) -> Path:
    path = repo / "reports" / "ledger" / "registry.json"
    _write_json(
        path,
        {
            "schema_version": ledger.REGISTRY_SCHEMA_VERSION,
            "benchmark": {
                "rows": 274,
                "sha256": ledger.DEFAULT_BENCHMARK_SHA256,
            },
            "judge": {
                "lineage": ledger.DEFAULT_JUDGE_LINEAGE,
                "image_judge_rows": 97,
            },
            "branches": [
                {
                    "id": "candidate",
                    "label": "Candidate",
                    "preregistered": True,
                    "report_globs": globs or ["reports/run/**/matched_score.json"],
                }
            ],
        },
    )
    return path


def _valid_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path
    judge_path = repo / "reports" / "run" / "evaluation" / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    report_path = repo / "reports" / "run" / "evaluation" / "matched_score.json"
    _write_json(report_path, _matched_report(judge_path))
    return repo, _registry(repo), report_path


def _mark_superseded(
    repo: Path,
    registry_path: Path,
    *,
    registry_call_count: int = 0,
    attested_call_count: int = 0,
    corrupt_hash: bool = False,
) -> Path:
    attestation_path = (
        repo / "reports" / "superseded" / "superseded_before_calls_attestation.json"
    )
    _write_json(
        attestation_path,
        {
            "schema_version": "synthetic-supersession-attestation-v1",
            "status": "SUPERSEDED_BEFORE_CALLS",
            "server_evidence": {"source_call_count": attested_call_count},
        },
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["branches"][0].update(
        {
            "superseded_before_calls": True,
            "superseded_source_call_count": registry_call_count,
            "supersession_attestation": str(
                attestation_path.relative_to(repo).as_posix()
            ),
            "supersession_attestation_sha256": (
                "0" * 64 if corrupt_hash else _sha(attestation_path)
            ),
        }
    )
    _write_json(registry_path, registry)
    return attestation_path


def _write_canonical_finalization(score_path: Path, *, corrupt_score_hash: bool = False) -> None:
    manifest_path = score_path.parent / "finalization_manifest.json"
    score_sha = "0" * 64 if corrupt_score_hash else _sha(score_path)
    _write_json(
        manifest_path,
        {
            "schema_version": "maxim-online-finalization-v1",
            "frozen_judge_v2": {
                "prompt_version": ledger.EXPECTED_JUDGE_PROMPT,
                "backend_config": ledger.FROZEN_JUDGE_BACKEND_CONFIG,
                "backend_config_sha256": ledger.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
                "adapter_sha256": ledger.FROZEN_JUDGE_ADAPTER_SHA256,
            },
            "score": {"output_hashes": {"json": score_sha}},
        },
    )
    state_path = score_path.parent / "postgeneration_orchestration_v1.json"
    sources = {
        "benchmark": {
            "sha256": ledger.DEFAULT_BENCHMARK_SHA256,
            "rows": ledger.DEFAULT_BENCHMARK_ROWS,
        },
        "baseline_solver": {
            "sha256": ledger.FROZEN_BASELINE_SOLVER_SHA256,
            "rows": ledger.DEFAULT_BENCHMARK_ROWS,
        },
        "baseline_judge": {
            "sha256": ledger.FROZEN_BASELINE_JUDGE_SHA256,
            "rows": ledger.DEFAULT_BENCHMARK_ROWS,
        },
        "image_template": {
            "sha256": ledger.FROZEN_IMAGE_TEMPLATE_SHA256,
            "rows": ledger.EXPECTED_IMAGE_JUDGE_ROWS,
        },
    }
    _write_json(
        state_path,
        {
            "schema_version": "full274-postgeneration-orchestration-v1",
            "sources": sources,
            "judge": {
                "model": ledger.EXPECTED_JUDGE_MODEL,
                "prompt_version": ledger.EXPECTED_JUDGE_PROMPT,
                "backend_config": ledger.FROZEN_JUDGE_BACKEND_CONFIG,
                "backend_config_sha256": ledger.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
            },
            "delegates": {
                name: {"sha256": digest}
                for name, digest in ledger.FROZEN_ORCHESTRATION_DELEGATE_SHA256.items()
            },
            "stages": {
                "prepare": {"status": "complete"},
                "judge": {"status": "complete"},
                "finalize": {
                    "status": "complete",
                    "artifacts": {
                        "score_json": {"sha256": _sha(score_path)},
                        "finalization_manifest": {"sha256": _sha(manifest_path)},
                    },
                },
            },
        },
    )
    state_path.with_suffix(".sha256").write_text(
        f"{_sha(state_path)}  {state_path.name}\n", encoding="utf-8"
    )


def test_accepts_exact_matched_full274_and_recomputes_accuracy(tmp_path: Path) -> None:
    repo, registry, report = _valid_fixture(tmp_path)
    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "final"
    assert row["correct"] == 192
    assert row["denominator"] == 274
    assert row["accuracy"] == pytest.approx(192 / 274)
    assert row["report"]["sha256"] == _sha(report)
    assert row["lineage_verification"]["lineage"] == ledger.DEFAULT_JUDGE_LINEAGE


def test_preregistered_branch_without_report_is_pending(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    result = ledger.build_ledger(tmp_path, registry)
    assert result["branches"][0]["status"] == "pending"
    assert result["summary"]["pending"] == 1


def test_valid_superseded_before_calls_is_non_final_without_metric(
    tmp_path: Path,
) -> None:
    repo, registry, report_path = _valid_fixture(tmp_path)
    attestation_path = _mark_superseded(repo, registry)
    # A superseded branch must not inspect or accept even a matching score path.
    report_path.write_text("candidate score must not be opened", encoding="utf-8")

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "non_final"
    assert row["result_kind"] == "superseded_before_calls"
    assert row["terminal_state"] == "SUPERSEDED_BEFORE_CALLS"
    assert row["source_call_count"] == 0
    assert row["supersession_attestation"] == {
        "path": attestation_path.relative_to(repo).as_posix(),
        "sha256": _sha(attestation_path),
        "schema_version": "synthetic-supersession-attestation-v1",
        "status": "SUPERSEDED_BEFORE_CALLS",
    }
    assert not {"correct", "denominator", "accuracy", "report"} & set(row)
    assert result["summary"]["non_final"] == 1
    assert result["summary"]["pending"] == 0
    assert result["summary"]["final"] == 0
    rendered = ledger.render_markdown(result)
    assert "| Candidate | non_final | — | — | — |" in rendered


def test_superseded_registry_source_call_count_must_be_exact_zero(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    _mark_superseded(tmp_path, registry, registry_call_count=1)
    with pytest.raises(ledger.LedgerError, match="must be exactly 0"):
        ledger.build_ledger(tmp_path, registry)


def test_superseded_attestation_hash_must_match(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _mark_superseded(tmp_path, registry, corrupt_hash=True)
    with pytest.raises(ledger.LedgerError, match="attestation SHA256 mismatch"):
        ledger.build_ledger(tmp_path, registry)


def test_superseded_attested_source_call_count_must_be_exact_zero(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    _mark_superseded(tmp_path, registry, attested_call_count=1)
    with pytest.raises(
        ledger.LedgerError,
        match=r"server_evidence\.source_call_count must be exactly 0",
    ):
        ledger.build_ledger(tmp_path, registry)


def test_superseded_attestation_path_must_stay_under_reports(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _mark_superseded(tmp_path, registry)
    value = json.loads(registry.read_text(encoding="utf-8"))
    value["branches"][0]["supersession_attestation"] = "../outside.json"
    _write_json(registry, value)
    with pytest.raises(ledger.LedgerError, match="unsafe .*supersession_attestation"):
        ledger.build_ledger(tmp_path, registry)


def test_wrong_benchmark_sha_is_rejected_not_final(tmp_path: Path) -> None:
    repo, registry, report_path = _valid_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"]["benchmark"]["sha256"] = "0" * 64
    _write_json(report_path, report)
    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "benchmark SHA256 mismatch" in row["rejected_candidates"][0]["reason"]


def test_wrong_judge_seed_is_rejected(tmp_path: Path) -> None:
    repo, registry, report_path = _valid_fixture(tmp_path)
    judge_path = report_path.parent / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows(wrong_seed=True))
    report = _matched_report(judge_path)
    _write_json(report_path, report)
    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "wrong seed" in row["rejected_candidates"][0]["reason"]


def test_partial_task_outcomes_are_rejected(tmp_path: Path) -> None:
    repo, registry, report_path = _valid_fixture(tmp_path)
    judge_path = report_path.parent / "image_judge.jsonl"
    _write_json(report_path, _matched_report(judge_path, correct=192, rows=273))
    result = ledger.build_ledger(repo, registry)
    assert result["branches"][0]["status"] == "pending"
    assert "must contain 274 rows" in result["branches"][0]["rejected_candidates"][0]["reason"]


def test_bounds_only_and_in_sample_reports_never_become_final(tmp_path: Path) -> None:
    repo, registry, report_path = _valid_fixture(tmp_path)
    judge_path = report_path.parent / "image_judge.jsonl"
    report = _matched_report(judge_path)
    report["overall"]["lower_correct"] = 180
    report["overall"]["upper_correct"] = 200
    report["evaluation_mode"] = "in_sample"
    _write_json(report_path, report)
    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "non-final" in row["rejected_candidates"][0]["reason"]


def test_finalized_score_requires_hash_valid_manifest(tmp_path: Path) -> None:
    repo = tmp_path
    judge_path = repo / "reports" / "run" / "finalized" / "raw" / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = judge_path.parent / "score.json"
    score = _matched_report(judge_path)
    score["judge"] = {"lineage": ledger.DEFAULT_JUDGE_LINEAGE, "matched": True}
    _write_json(score_path, score)
    registry = _registry(repo, ["reports/run/finalized/**/score.json"])

    first = ledger.build_ledger(repo, registry)
    assert first["branches"][0]["status"] == "pending"
    assert "finalization_manifest" in first["branches"][0]["rejected_candidates"][0]["reason"]

    _write_json(
        score_path.parent / "finalization_manifest.json",
        {"score": {"path": str(score_path), "sha256": _sha(score_path)}},
    )
    second = ledger.build_ledger(repo, registry)
    assert second["branches"][0]["status"] == "final"


def test_accepts_canonical_evaluation_score_with_exact_orchestration(tmp_path: Path) -> None:
    repo = tmp_path
    output_dir = repo / "reports" / "run" / "arbitrary_output_dir"
    judge_path = output_dir / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = output_dir / "score.json"
    _write_json(score_path, _matched_report(judge_path, correct=173))
    _write_canonical_finalization(score_path)
    registry = _registry(repo, ["reports/run/arbitrary_output_dir/score.json"])

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "final"
    assert row["correct"] == 173
    assert row["finalization_manifest"]["orchestration"][
        "frozen_lineage_verified"
    ] is True


def test_canonical_evaluation_score_without_manifest_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path
    output_dir = repo / "reports" / "run" / "evaluation"
    judge_path = output_dir / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = output_dir / "score.json"
    _write_json(score_path, _matched_report(judge_path))
    registry = _registry(repo, ["reports/run/evaluation/score.json"])

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "has no finalization_manifest.json" in row["rejected_candidates"][0]["reason"]


def test_canonical_evaluation_score_with_bad_manifest_hash_is_rejected(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output_dir = repo / "reports" / "run" / "evaluation"
    judge_path = output_dir / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = output_dir / "score.json"
    _write_json(score_path, _matched_report(judge_path))
    _write_canonical_finalization(score_path, corrupt_score_hash=True)
    registry = _registry(repo, ["reports/run/evaluation/score.json"])

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "score SHA256 mismatch" in row["rejected_candidates"][0]["reason"]


def test_canonical_evaluation_score_without_orchestration_is_rejected(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output_dir = repo / "reports" / "run" / "evaluation"
    judge_path = output_dir / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = output_dir / "score.json"
    _write_json(score_path, _matched_report(judge_path))
    _write_canonical_finalization(score_path)
    (output_dir / "postgeneration_orchestration_v1.json").unlink()
    (output_dir / "postgeneration_orchestration_v1.sha256").unlink()
    registry = _registry(repo, ["reports/run/evaluation/score.json"])

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "has no postgeneration_orchestration" in row["rejected_candidates"][0]["reason"]


def test_canonical_evaluation_score_with_tampered_orchestration_is_rejected(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output_dir = repo / "reports" / "run" / "evaluation"
    judge_path = output_dir / "image_judge.jsonl"
    _write_jsonl(judge_path, _judge_rows())
    score_path = output_dir / "score.json"
    _write_json(score_path, _matched_report(judge_path))
    _write_canonical_finalization(score_path)
    state_path = output_dir / "postgeneration_orchestration_v1.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["judge"]["backend_config"]["seed"] = 7
    _write_json(state_path, state)
    state_path.with_suffix(".sha256").write_text(
        f"{_sha(state_path)}  {state_path.name}\n", encoding="utf-8"
    )
    registry = _registry(repo, ["reports/run/evaluation/score.json"])

    result = ledger.build_ledger(repo, registry)
    row = result["branches"][0]
    assert row["status"] == "pending"
    assert "frozen judge lineage mismatch" in row["rejected_candidates"][0]["reason"]


def test_multiple_valid_reports_are_conflict_not_silently_selected(tmp_path: Path) -> None:
    repo, _, first_path = _valid_fixture(tmp_path)
    second_dir = repo / "reports" / "run" / "rerun"
    second_dir.mkdir(parents=True)
    second_judge = second_dir / "image_judge.jsonl"
    _write_jsonl(second_judge, _judge_rows())
    _write_json(second_dir / "matched_score.json", _matched_report(second_judge))
    registry = _registry(repo, ["reports/run/**/matched_score.json"])
    result = ledger.build_ledger(repo, registry)
    assert first_path.is_file()
    assert result["branches"][0]["status"] == "conflict"
    assert len(result["branches"][0]["accepted_candidates"]) == 2


def test_markdown_never_formats_rejected_bound_as_score(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    result = ledger.build_ledger(tmp_path, registry)
    rendered = ledger.render_markdown(result)
    assert "| Candidate | pending | \u2014 | \u2014 | \u2014 |" in rendered
    assert "bounds, partial/interim, in-sample, and OOF estimates are never shown as final" in rendered


def test_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    repo, registry, _ = _valid_fixture(tmp_path)
    result = ledger.build_ledger(repo, registry)
    out_json = repo / "reports" / "ledger" / "RESULTS.json"
    out_md = repo / "reports" / "ledger" / "RESULTS.md"
    ledger.write_outputs(result, out_json, out_md)
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["summary"]["final"] == 1
    assert loaded["branches"][0]["correct"] == 192
    assert "192/274" in out_md.read_text(encoding="utf-8")

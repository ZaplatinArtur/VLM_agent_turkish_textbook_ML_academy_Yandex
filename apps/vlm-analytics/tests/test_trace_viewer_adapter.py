from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_trace_viewer.adapter import (
    ArtifactError,
    FINAL_COMPOSED,
    FINAL_EVALUATION,
    FINAL_RESOLVER,
    POST_SCORE,
    SPEED_ANALYSIS,
    V7ArtifactAdapter,
    discover_artifact_root,
)
from vlm_trace_viewer.model import split_solution_steps


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value, ensure_ascii=False) for value in values) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path) -> Path:
    _json(
        root / SPEED_ANALYSIS,
        {
            "rows": 2,
            "source_shortcuts": 1,
            "anchor_fallbacks": 1,
            "source_shortcut_rate": 0.5,
            "answer_equivalent_rows": 1,
            "recorded_anchor_usage": {
                "avoidable_latency_fraction": 0.4,
                "avoidable_input_fraction": 0.45,
                "avoidable_output_fraction": 0.35,
            },
            "claims": {
                "online_wall_clock_speedup_measured": False,
                "source_lookup_cost_included": False,
            },
        },
    )
    _json(
        root / POST_SCORE,
        {
            "comparison_to_v6": {},
            "changed_correctness_vs_v6": [
                {"task_id": "t1", "mechanism": "official-source answer replacement"}
            ],
            "limitations": ["development replay"],
        },
    )
    _jsonl(
        root / FINAL_COMPOSED / "solver.jsonl",
        [
            {
                "task_id": "t1",
                "final_answer": "B",
                "reasoning": "source answer",
                "solution_steps": "1. inspect page\n2. bind key",
                "model": "demo-model",
                "prompt_version": "demo",
                "usage": {"latency_s": 2.0, "input_tokens": 10, "output_tokens": 4},
            },
            {
                "task_id": "t2",
                "final_answer": "A",
                "reasoning": "anchor answer",
                "solution_steps": "keep anchor",
                "model": "demo-model",
                "prompt_version": "demo",
                "usage": {"latency_s": 4.0, "input_tokens": 11, "output_tokens": 5},
            },
        ],
    )
    _jsonl(
        root / FINAL_COMPOSED / "decisions.jsonl",
        [
            {
                "task_id": "t1",
                "source_override": True,
                "certificate_trace_fingerprint": "trace-1",
            },
            {"task_id": "t2", "anchor_bytes_copied": True},
        ],
    )
    _jsonl(
        root / FINAL_RESOLVER / "candidate.jsonl",
        [
            {"task_id": "t1", "abstain": False, "final_answer": "B"},
            {"task_id": "t2", "abstain": True, "final_answer": ""},
        ],
    )
    _jsonl(
        root / FINAL_RESOLVER / "certificates.jsonl",
        [
            {
                "task_id": "t1",
                "status": "pass",
                "strength": "strong",
                "trace_fingerprint": "trace-1",
                "verifier": "fixture-verifier",
                "trace": {
                    "checks": {"question_binding": True, "valid_source_answer": True},
                    "match": {"page_idf_coverage": 0.9, "page_margin": 0.4},
                    "source": {
                        "document_id": "book-1",
                        "name": "book.pdf",
                        "public_locator": "https://example.test/book.pdf",
                        "matched_page_number": 4,
                        "key_page_number": 10,
                        "question_number": 2,
                        "record_id": "book-1:p4:q2",
                        "key_bbox": [1, 2, 3, 4],
                        "pdf_sha256": "abc",
                    },
                },
            }
        ],
    )
    _json(
        root / FINAL_EVALUATION / "score.json",
        {
            "overall": {
                "n": 2,
                "new_correct": 1,
                "new_accuracy": 0.5,
                "baseline_accuracy": 0.0,
            },
            "by_subject": {
                "Math": {"n": 2, "new_correct": 1, "new_accuracy": 0.5}
            },
            "operational": {
                "latency": {
                    "latency_s_median": 3.0,
                    "latency_s_p95_nearest_rank": 4.0,
                    "latency_s_max": 4.0,
                }
            },
            "task_outcomes": [
                {
                    "task_id": "t1",
                    "subject": "Math",
                    "answer_type": "choice",
                    "new_correct": True,
                    "baseline_correct": False,
                    "score_source": "deterministic",
                    "score_method": "choice",
                    "transition": "fixed",
                },
                {
                    "task_id": "t2",
                    "subject": "Math",
                    "answer_type": "choice",
                    "new_correct": False,
                    "baseline_correct": False,
                    "score_source": "deterministic",
                    "score_method": "choice",
                    "transition": "both_wrong",
                },
            ],
        },
    )
    return root


def test_adapter_joins_v7_and_normalizes_layered_override(tmp_path: Path) -> None:
    dataset = V7ArtifactAdapter(_fixture(tmp_path)).load()

    assert dataset.summary.rows == 2
    assert dataset.summary.correct == 1
    assert dataset.summary.source_certificates == 1
    assert dataset.summary.answer_overrides == 1
    assert dataset.summary.latency_median_s == 3.0
    assert dataset.summary.source_shortcuts == 1
    assert dataset.summary.source_shortcut_rate == 0.5
    assert dataset.summary.answer_equivalent_shortcuts == 1
    assert dataset.summary.avoidable_recorded_latency_fraction == 0.4
    assert dataset.summary.speed_online_wall_clock_measured is False

    overridden = next(task for task in dataset.tasks if task.task_id == "t1")
    assert overridden.decision_action == "replace_anchor"
    assert overridden.challenger_answer == "B"
    assert overridden.source.accepted is True
    assert overridden.source.matched_page == 4
    assert overridden.pipeline[4].state == "pass"

    fallback = next(task for task in dataset.tasks if task.task_id == "t2")
    assert fallback.decision_action == "keep_anchor"
    assert fallback.source.accepted is False
    assert fallback.pipeline[2].state == "skipped"


def test_split_solution_steps_keeps_content_and_removes_markers() -> None:
    assert split_solution_steps("1. first\n- second\n\n3. third") == (
        "first",
        "second",
        "third",
    )


def test_adapter_rejects_speed_analysis_for_a_different_row_set(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    speed = json.loads((root / SPEED_ANALYSIS).read_text(encoding="utf-8"))
    speed["rows"] = 3
    speed["anchor_fallbacks"] = 2
    _json(root / SPEED_ANALYSIS, speed)

    with pytest.raises(ArtifactError, match="row count differs"):
        V7ArtifactAdapter(root).load()


def test_discovery_finds_sibling_from_nested_packaged_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    app = workspace / "target-repo" / "apps" / "vlm-analytics"
    artifact_root = workspace / "VLM_agent_turkish_textbook_basic_rag"
    app.mkdir(parents=True)
    _json(artifact_root / POST_SCORE, {"schema_version": "fixture"})
    monkeypatch.chdir(app)
    monkeypatch.delenv("VLM_TRACE_ARTIFACT_ROOT", raising=False)

    assert discover_artifact_root() == artifact_root.resolve()


def test_explicit_artifact_root_fails_instead_of_silently_falling_back(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArtifactError, match="explicit --artifact-root"):
        discover_artifact_root(tmp_path / "wrong-project")

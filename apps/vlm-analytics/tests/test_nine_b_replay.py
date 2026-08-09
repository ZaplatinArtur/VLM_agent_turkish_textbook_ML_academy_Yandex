from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vlm_trace_viewer.adapter import ArtifactError
from vlm_trace_viewer.nine_b_adapter import NineBV7ArtifactAdapter
from vlm_trace_viewer.replay_aggregate import (
    AGGREGATE_SCHEMA,
    COMPARISON_SCHEMA,
    EMPTY_UNION_SHA256,
    EXPECTED_MODEL,
    MILESTONE_SPECS,
    NORMALIZED_V2_ADAPTER,
    SCORE_SCHEMA,
    ReplayAggregateError,
    _resolve_descriptor,
    _resolve_native_descriptor,
    _validate_native_closures,
    _validate_native_judge_manifest,
    load_frozen_9b_comparison,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _role_descriptor(root: Path, path: Path, role: str) -> dict[str, str]:
    return {"role": role, **_descriptor(root, path)}


def _rewrite_pinned_aggregate(
    manifest: Path,
    milestone_id: str,
    mutate,
) -> None:
    comparison = json.loads(manifest.read_text(encoding="utf-8"))
    descriptor = next(
        item for item in comparison["milestones"] if item["milestone_id"] == milestone_id
    )["aggregate"]
    unresolved = Path(descriptor["path"])
    aggregate_path = (
        unresolved.resolve()
        if unresolved.is_absolute()
        else (manifest.parent / unresolved).resolve()
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    mutate(aggregate)
    _json(aggregate_path, aggregate)
    descriptor["sha256"] = _sha(aggregate_path)
    _json(manifest, comparison)


def _metrics() -> dict:
    return {
        "rows": 2,
        "correct": 1,
        "accuracy": 0.5,
        "slices": {
            "deterministic": {"rows": 1, "correct": 1, "accuracy": 1.0},
            "image": {"rows": 1, "correct": 0, "accuracy": 0.0},
            "math": {"rows": 1, "correct": 1, "accuracy": 1.0},
            "non_math": {"rows": 1, "correct": 0, "accuracy": 0.0},
        },
    }


def _build_comparison(
    root: Path,
    *,
    foreign_model_milestone: str | None = None,
    omit_page_caveat: bool = False,
    with_question_images: bool = False,
) -> Path:
    benchmark = root / "benchmark.jsonl"
    benchmark_rows = [
        {
            "task_id": "t1",
            "subject": "Math",
            "grade": "12",
            "answer_type": "choice",
            "question": "q1",
        },
        {
            "task_id": "t2",
            "subject": "Science",
            "grade": "9",
            "answer_type": "choice",
            "question": "q2",
        },
    ]
    if with_question_images:
        image_root = root / "tmp" / "blind_visual_binding" / "task_images"
        for row in benchmark_rows:
            task_id = row["task_id"]
            row["question_images"] = [
                {"format": "file_path", "data": f"images/{task_id}.png"}
            ]
            image = image_root / f"{task_id}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"display-only fixture")
    _jsonl(benchmark, benchmark_rows)
    benchmark_hash = _sha(benchmark)
    milestone_descriptors: list[dict] = []

    for milestone_id, _, pipeline, provenance_status in MILESTONE_SPECS:
        directory = root / milestone_id
        is_source = milestone_id.startswith("source_")
        solver = directory / "solver.jsonl"
        solver_model = (
            "Qwen/Qwen3.5-27B"
            if milestone_id == foreign_model_milestone
            else EXPECTED_MODEL
        )
        solver_rows = [
            {
                "task_id": "t1",
                "model": solver_model,
                "final_origin": (
                    "deterministic_source_replacement" if is_source else "model_anchor"
                ),
                "anchor_answer": "A",
                "final_answer": "B" if is_source else "A",
                "reasoning": "saved anchor reasoning",
                "solution_steps": "1. inspect",
                "usage": {"latency_s": 2.0, "input_tokens": 10, "output_tokens": 4},
            },
            {
                "task_id": "t2",
                "model": EXPECTED_MODEL,
                "final_origin": "model_anchor",
                "anchor_answer": "A",
                "final_answer": "A",
                "reasoning": "saved anchor reasoning",
                "usage": {"latency_s": 4.0, "input_tokens": 11, "output_tokens": 5},
            },
        ]
        _jsonl(solver, solver_rows)
        raw_solver = directory / "raw_solver.jsonl"
        _jsonl(
            raw_solver,
            [
                {key: value for key, value in row.items() if key != "final_origin"}
                for row in solver_rows
            ],
        )
        judge = directory / "judge.jsonl"
        _jsonl(
            judge,
            [
                {"task_id": "t1", "correct": True, "subject": "Math", "score_source": "det"},
                {"task_id": "t2", "correct": False, "subject": "Science", "score_source": "judge"},
            ],
        )
        certificates: list[dict] = []
        certificate_hashes: list[str] = []
        if is_source:
            certificate = directory / "certificates.jsonl"
            _jsonl(
                certificate,
                [{"task_id": "t1", "status": "pass", "strength": "strong"}],
            )
            certificates.append(_role_descriptor(root, certificate, "source_union"))
            certificate_hashes.append(_sha(certificate))

        provenance_manifests: list[dict] = []
        if provenance_status != "matched_judge_replay_partial_generation_provenance":
            provenance = directory / "provenance.json"
            _json(provenance, {"milestone_id": milestone_id, "status": provenance_status})
            provenance_manifests.append(
                _role_descriptor(root, provenance, "profile_bound_before_score")
            )
        bound_before_score = (
            True
            if provenance_status in {"preregistered_gold_blind", "new_profile_bound_replay"}
            else None
        )
        caveats = (
            []
            if provenance_status in {"preregistered_gold_blind", "new_profile_bound_replay"}
            else ["legacy provenance is weaker than the new profile-bound replay"]
        )
        if milestone_id == "page_rag_9b" and omit_page_caveat:
            caveats = []
        source_union = {
            "sha256": hashlib.sha256(b"t1\n").hexdigest() if is_source else EMPTY_UNION_SHA256,
            "size": 1 if is_source else 0,
            "replacements": 1 if is_source else 0,
            "confirmations": 0,
            "stage_counts": {milestone_id: 1} if is_source else {},
        }
        origins = {
            "model_anchor": 1 if is_source else 2,
            "deterministic_source_replacement": 1 if is_source else 0,
            "unknown": 0,
        }
        foreign_models = [solver_model] if solver_model != EXPECTED_MODEL else []
        model_closure = {
            "expected_model": EXPECTED_MODEL,
            "checked_rows": 2,
            "matching_rows": 1 if foreign_models else 2,
            "foreign_models": foreign_models,
        }
        evaluator = {
            "semantics": "deterministic_plus_image_judge_v1",
            "deterministic_rows": 1,
            "image_rows": 1,
            "source_certified_image_rows": 1 if is_source else 0,
            "model_judged_image_rows": 0 if is_source else 1,
            "judge_model": EXPECTED_MODEL,
        }
        comparisons: list[dict] = []
        score_value = {
            "schema_version": SCORE_SCHEMA,
            "milestone_id": milestone_id,
            "model": EXPECTED_MODEL,
            "pipeline": pipeline,
            "benchmark_sha256": benchmark_hash,
            "solver_sha256": _sha(solver),
            "judge_sha256": _sha(judge),
            "certificate_sha256s": certificate_hashes,
            "metrics": _metrics(),
            "source_union": source_union,
            "comparisons": comparisons,
            "evaluator": evaluator,
            "final_origin_counts": origins,
        }
        score = directory / "score.json"
        _json(score, score_value)
        aggregate_value = {
            "schema_version": AGGREGATE_SCHEMA,
            "milestone_id": milestone_id,
            "model": EXPECTED_MODEL,
            "pipeline": pipeline,
            "provenance_status": provenance_status,
            "bound_before_score": bound_before_score,
            "caveats": caveats,
            "provenance_manifests": provenance_manifests,
            "artifacts": {
                "solver": _descriptor(root, solver),
                "raw_solver": _descriptor(root, raw_solver),
                "score": _descriptor(root, score),
                "judge": _descriptor(root, judge),
                "certificates": certificates,
            },
            "certificate_absence_reason": None if is_source else "non-source milestone",
            "benchmark_sha256": benchmark_hash,
            "metrics": _metrics(),
            "model_closure": model_closure,
            "source_union": source_union,
            "comparisons": comparisons,
            "evaluator": evaluator,
            "final_origin_counts": origins,
        }
        aggregate = directory / "aggregate.json"
        _json(aggregate, aggregate_value)
        milestone_descriptors.append(
            {
                "milestone_id": milestone_id,
                "adapter": NORMALIZED_V2_ADAPTER,
                "aggregate": _descriptor(root, aggregate),
            }
        )

    comparison = root / "comparison.json"
    _json(
        comparison,
        {
            "schema_version": COMPARISON_SCHEMA,
            "model": EXPECTED_MODEL,
            "benchmark": _descriptor(root, benchmark),
            "milestones": milestone_descriptors,
        },
    )
    return comparison


def test_seven_stage_loader_and_final_trace_adapter(tmp_path: Path) -> None:
    comparison = load_frozen_9b_comparison(_build_comparison(tmp_path))

    assert len(comparison.milestones) == 7
    assert comparison.final.milestone_id == "source_v7_rebase_9b"
    assert comparison.final.final_origin_counts == {
        "model_anchor": 1,
        "deterministic_source_replacement": 1,
        "unknown": 0,
    }
    assert comparison.milestones[0].provenance_status == "historical_output_control"
    assert comparison.milestones[1].bound_before_score is None

    dataset = NineBV7ArtifactAdapter(comparison).load()
    assert dataset.summary.rows == 2
    assert dataset.summary.pipeline_provenance.startswith("9B Query Active Crop V2")
    replaced = next(task for task in dataset.tasks if task.task_id == "t1")
    assert replaced.base_row_model == EXPECTED_MODEL
    assert replaced.final_origin == "deterministic source-adjudicated replacement"
    assert replaced.source.accepted is True


def test_loader_rejects_foreign_27b_solver_row(tmp_path: Path) -> None:
    manifest = _build_comparison(
        tmp_path,
        foreign_model_milestone="source_v7_rebase_9b",
    )
    with pytest.raises(ReplayAggregateError, match="model closure"):
        load_frozen_9b_comparison(manifest)


def test_loader_requires_explicit_legacy_caveat(tmp_path: Path) -> None:
    manifest = _build_comparison(tmp_path, omit_page_caveat=True)
    with pytest.raises(ReplayAggregateError, match="requires an explicit caveat"):
        load_frozen_9b_comparison(manifest)


def test_comparison_dispatch_and_inner_hashes_fail_closed(tmp_path: Path) -> None:
    manifest = _build_comparison(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["milestones"][3]["adapter"] = "unknown_native_schema"
    _json(manifest, value)
    with pytest.raises(ReplayAggregateError, match="unsupported native aggregate adapter"):
        load_frozen_9b_comparison(manifest)

    manifest = _build_comparison(tmp_path / "hash_case")
    solver = tmp_path / "hash_case" / "source_v7_rebase_9b" / "solver.jsonl"
    solver.write_text(solver.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReplayAggregateError, match="SHA-256 mismatch"):
        load_frozen_9b_comparison(manifest)


def test_normalized_raw_solver_descriptor_is_required_and_verified(tmp_path: Path) -> None:
    missing_manifest = _build_comparison(tmp_path / "missing")
    _rewrite_pinned_aggregate(
        missing_manifest,
        "query_active_crop_v2_9b",
        lambda aggregate: aggregate["artifacts"].pop("raw_solver"),
    )
    with pytest.raises(ReplayAggregateError, match=r"missing=\['raw_solver'\]"):
        load_frozen_9b_comparison(missing_manifest)

    wrong_hash_manifest = _build_comparison(tmp_path / "wrong_hash")
    _rewrite_pinned_aggregate(
        wrong_hash_manifest,
        "query_active_crop_v2_9b",
        lambda aggregate: aggregate["artifacts"]["raw_solver"].update(
            {"sha256": "0" * 64}
        ),
    )
    with pytest.raises(ReplayAggregateError, match="raw_solver SHA-256 mismatch"):
        load_frozen_9b_comparison(wrong_hash_manifest)

    wrong_path_manifest = _build_comparison(tmp_path / "wrong_path")

    def point_raw_alias_at_judge(aggregate: dict) -> None:
        aggregate["artifacts"]["raw_solver"] = dict(aggregate["artifacts"]["judge"])

    _rewrite_pinned_aggregate(
        wrong_path_manifest,
        "query_active_crop_v2_9b",
        point_raw_alias_at_judge,
    )
    with pytest.raises(ReplayAggregateError, match="exact final_origin-only"):
        load_frozen_9b_comparison(wrong_path_manifest)


def test_missing_absolute_descriptor_rebases_inside_current_clone(tmp_path: Path) -> None:
    clone = tmp_path / "new_clone"
    base = clone / "reports" / "frozen_run"
    target = clone / "artifacts" / "payload.json"
    _json(target, {"value": "portable"})
    base.mkdir(parents=True)
    stale = tmp_path / "retired_checkout" / "artifacts" / "payload.json"
    descriptor = {"path": str(stale), "sha256": _sha(target)}

    path, digest = _resolve_descriptor(base, descriptor, "portable fixture")

    assert path == target.resolve()
    assert digest == _sha(target)

    native = clone / "reports" / "rows.jsonl"
    _jsonl(native, [{"task_id": "t1"}])
    native_stale = tmp_path / "retired_checkout" / "reports" / "rows.jsonl"
    path, digest, rows = _resolve_native_descriptor(
        base,
        {"path": str(native_stale), "sha256": _sha(native), "rows": 1},
        "portable native fixture",
        rows_required=True,
    )
    assert (path, digest, rows) == (native.resolve(), _sha(native), 1)


def test_portable_rebase_fails_closed_on_tamper_traversal_and_ambiguity(
    tmp_path: Path,
) -> None:
    clone = tmp_path / "clone"
    base = clone / "reports" / "run"
    target = clone / "artifacts" / "payload.json"
    _json(target, {"value": "expected"})
    base.mkdir(parents=True)
    stale = tmp_path / "retired" / "artifacts" / "payload.json"
    expected = _sha(target)
    target.write_text('{"value":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ReplayAggregateError, match="SHA-256 mismatch"):
        _resolve_descriptor(
            base,
            {"path": str(stale), "sha256": expected},
            "tampered portable fixture",
        )

    existing_original = tmp_path / "retired" / "artifacts" / "preferred.json"
    clone_copy = clone / "artifacts" / "preferred.json"
    _json(existing_original, {"authority": "existing absolute path"})
    _json(clone_copy, {"authority": "portable fallback"})
    with pytest.raises(ReplayAggregateError, match="SHA-256 mismatch"):
        _resolve_descriptor(
            base,
            {"path": str(existing_original), "sha256": _sha(clone_copy)},
            "existing absolute fixture",
        )

    unknown_prefix = tmp_path / "retired" / "private_outputs" / "payload.json"
    with pytest.raises(ReplayAggregateError, match="unknown or ambiguous portable prefix"):
        _resolve_descriptor(
            base,
            {"path": str(unknown_prefix), "sha256": "0" * 64},
            "unknown prefix fixture",
        )

    traversal = tmp_path / "retired" / "artifacts" / ".." / "outside.json"
    with pytest.raises(ReplayAggregateError, match="traversal"):
        _resolve_descriptor(
            base,
            {"path": str(traversal), "sha256": "0" * 64},
            "traversal fixture",
        )

    outer = tmp_path / "ambiguous"
    inner = outer / "reports" / "nested"
    ambiguous_base = inner / "reports" / "run"
    first = outer / "reports" / "same.json"
    second = inner / "reports" / "same.json"
    _json(first, {"same": True})
    _json(second, {"same": True})
    ambiguous_base.mkdir(parents=True)
    ambiguous_stale = tmp_path / "retired" / "reports" / "same.json"
    with pytest.raises(ReplayAggregateError, match="ambiguous"):
        _resolve_descriptor(
            ambiguous_base,
            {"path": str(ambiguous_stale), "sha256": _sha(first)},
            "ambiguous fixture",
        )


def test_9b_display_images_are_bounded_and_do_not_import_archived_rows(
    tmp_path: Path,
) -> None:
    comparison = load_frozen_9b_comparison(
        _build_comparison(tmp_path, with_question_images=True)
    )
    adapter = NineBV7ArtifactAdapter(comparison, display_asset_root=tmp_path)
    dataset = adapter.load()

    assert sum(task.question_image is not None for task in dataset.tasks) == 2
    assert all(
        task.question_image is not None
        and task.question_image.is_relative_to(tmp_path.resolve())
        for task in dataset.tasks
    )
    outside = tmp_path.parent / "t1.png"
    outside.write_bytes(b"must never be used")
    assert adapter._resolve_question_image(  # noqa: SLF001 - security boundary test
        "t1", {"question_images": [{"data": "../../t1.png"}]}
    ) is None

    with pytest.raises(ArtifactError, match="does not contain the verified 9B benchmark"):
        NineBV7ArtifactAdapter(comparison, display_asset_root=tmp_path / "elsewhere")


def test_native_model_and_answer_origin_closures_are_separate() -> None:
    generation, origins = _validate_native_closures(
        {
            "upstream_generation_model_closure": [EXPECTED_MODEL],
            "answer_origin_closure": [
                "deterministic_official_source_replacement",
                "official_source_confirmation_of_9b_anchor",
                "qwen35_9b_anchor_passthrough",
            ],
            "model_closure": [EXPECTED_MODEL],
            "final_origin_counts": {
                "deterministic_official_source_replacement": 1,
                "official_source_confirmation_of_9b_anchor": 1,
                "qwen35_9b_anchor_passthrough": 1,
            },
        }
    )
    assert generation == [EXPECTED_MODEL]
    assert origins == [
        "deterministic_official_source_replacement",
        "official_source_confirmation_of_9b_anchor",
        "qwen35_9b_anchor_passthrough",
    ]

    with pytest.raises(ReplayAggregateError, match="generation-only alias"):
        _validate_native_closures(
            {
                "upstream_generation_model_closure": [EXPECTED_MODEL],
                "answer_origin_closure": [
                    "deterministic_official_source_replacement",
                    "official_source_confirmation_of_9b_anchor",
                    "qwen35_9b_anchor_passthrough",
                ],
                "model_closure": [EXPECTED_MODEL, "deterministic_official_source"],
                "final_origin_counts": {
                    "deterministic_official_source_replacement": 1,
                    "official_source_confirmation_of_9b_anchor": 1,
                    "qwen35_9b_anchor_passthrough": 1,
                },
            }
        )


def test_native_judge_manifest_recounts_cumulative_lineage(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    output = tmp_path / "output.jsonl"
    base_solver = tmp_path / "base_solver.jsonl"
    composed_solver = tmp_path / "composed_solver.jsonl"
    _jsonl(
        base_solver,
        [
            {"task_id": "i1", "final_answer": "A"},
            {"task_id": "i2", "final_answer": "A"},
        ],
    )
    _jsonl(
        composed_solver,
        [
            {"task_id": "i1", "final_answer": "B"},
            {"task_id": "i2", "final_answer": "A"},
        ],
    )
    original_rows = [
        {
            "task_id": "i1",
            "judge": {"backend": "openai-compatible", "model": EXPECTED_MODEL},
            "verdict": {"strict_correct": False},
        },
        {
            "task_id": "i2",
            "judge": {"backend": "openai-compatible", "model": EXPECTED_MODEL},
            "verdict": {"strict_correct": True},
        },
    ]
    _jsonl(original, original_rows)
    _jsonl(
        output,
        [
            {
                "task_id": "i1",
                "judge": {
                    "backend": "deterministic-official-source-certificate",
                    "model": None,
                },
                "verdict": {"strict_correct": True},
            },
            original_rows[1],
        ],
    )
    manifest = tmp_path / "judge_manifest.json"
    value = {
        "base_image_judge": _descriptor(tmp_path, original),
        "base_solver": _descriptor(tmp_path, base_solver),
        "composition": {
            "solver": {**_descriptor(tmp_path, composed_solver), "rows": 2}
        },
        "output": {**_descriptor(tmp_path, output), "rows": 2},
        "source_adjudicated_image_rows": [
            {
                "task_id": "i1",
                "verdict_origin": "deterministic_official_source_adjudication",
                "stage_answer_action": "replace_immediate_base_with_source",
                "trace_fingerprint": "a" * 64,
            }
        ],
        "stage_source_adjudicated_image_rows_count": 1,
        "copied_base_judge_rows_byte_identical": 1,
        "cumulative_source_adjudicated_image_rows_count": 1,
        "cumulative_original_9b_judge_rows_count": 1,
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "inherited_27b_outputs": False,
        "upstream_generation_model_closure": [EXPECTED_MODEL],
    }
    _json(manifest, value)

    _, original_bytes, counts = _validate_native_judge_manifest(
        tmp_path,
        manifest,
        previous_output=None,
        original_9b_rows=None,
        image_rows=2,
        stage_index=0,
    )
    assert len(original_bytes) == 2
    assert counts == {
        "stage_source_adjudicated_image_rows_count": 1,
        "copied_base_judge_rows_byte_identical": 1,
        "cumulative_source_adjudicated_image_rows_count": 1,
        "cumulative_original_9b_judge_rows_count": 1,
    }

    value["copied_9b_judge_rows_byte_identical"] = value.pop(
        "copied_base_judge_rows_byte_identical"
    )
    _json(manifest, value)
    with pytest.raises(ReplayAggregateError, match="misleading"):
        _validate_native_judge_manifest(
            tmp_path,
            manifest,
            previous_output=None,
            original_9b_rows=None,
            image_rows=2,
            stage_index=0,
        )

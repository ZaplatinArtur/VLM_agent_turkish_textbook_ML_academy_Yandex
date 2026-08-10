from __future__ import annotations

import json
from pathlib import Path

from vlm_analytics.analytics import AnalyticsService
from vlm_analytics.chunking_importer import import_chunking_experiment
from vlm_analytics.database import Database


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_imports_chunking_experiment_and_metrics(tmp_path: Path) -> None:
    database = Database(tmp_path / "analytics.db")
    corpus = _write(
        tmp_path / "hybrid_chunking_all_test.json",
        {
            "books": 2,
            "pages": 10,
            "units": 30,
            "units_per_page": 3,
            "oversized_units": 1,
            "runtime_seconds": 0.5,
            "unit_kinds": {"theory": 20, "exercise": 8, "solution": 2},
        },
    )
    localization = _write(
        tmp_path / "chunk_localization_test.json",
        {
            "queries": 5,
            "page_chunking": {
                "hit_at_1": 0.2,
                "hit_at_5": 0.4,
                "mrr_at_5": 0.3,
            },
            "hybrid_task_chunking": {
                "hit_at_1": 0.8,
                "hit_at_5": 1.0,
                "mrr_at_5": 0.9,
            },
        },
    )
    refinement = _write(
        tmp_path / "hybrid_qwen_refine_test.json",
        {
            "successful_pages": 9,
            "selected_pages": 10,
            "refined_units": 20,
            "changed_units": 4,
            "change_rate": 0.2,
            "transitions": {},
        },
    )
    audit = _write(
        tmp_path / "hybrid_qwen_holdout_test_repaired.json",
        {
            "successful_pages": 10,
            "sampled_pages": 10,
            "audited_units": 40,
            "agreement_rate": 0.75,
        },
    )

    experiment_id = import_chunking_experiment(
        database,
        corpus_report=corpus,
        localization_report=localization,
        refinement_report=refinement,
        audit_report=audit,
    )
    experiment = AnalyticsService(database).latest_retrieval_experiment()

    assert experiment is not None
    assert experiment["id"] == experiment_id
    metrics = {
        item["metric_key"]: item for item in experiment["metrics"]
    }
    assert metrics["hit_at_1"]["baseline_value"] == 20
    assert metrics["hit_at_1"]["candidate_value"] == 80
    assert metrics["qwen_schema_success"]["candidate_value"] == 100


def test_reimport_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "analytics.db")
    paths = {
        "corpus_report": _write(
            tmp_path / "hybrid_chunking_all_test.json",
            {
                "books": 1,
                "pages": 1,
                "units": 2,
                "units_per_page": 2,
                "oversized_units": 0,
                "runtime_seconds": 0.1,
                "unit_kinds": {},
            },
        ),
        "localization_report": _write(
            tmp_path / "chunk_localization_test.json",
            {
                "queries": 1,
                "page_chunking": {
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "mrr_at_5": 0,
                },
                "hybrid_task_chunking": {
                    "hit_at_1": 1,
                    "hit_at_5": 1,
                    "mrr_at_5": 1,
                },
            },
        ),
        "refinement_report": _write(
            tmp_path / "hybrid_qwen_refine_test.json",
            {
                "successful_pages": 1,
                "selected_pages": 1,
                "refined_units": 1,
                "changed_units": 0,
                "change_rate": 0,
                "transitions": {},
            },
        ),
        "audit_report": _write(
            tmp_path / "hybrid_qwen_holdout_test_repaired.json",
            {
                "successful_pages": 1,
                "sampled_pages": 1,
                "audited_units": 1,
                "agreement_rate": 1,
            },
        ),
    }

    first = import_chunking_experiment(database, **paths)
    second = import_chunking_experiment(database, **paths)

    assert first == second
    assert database.scalar("SELECT COUNT(*) FROM retrieval_experiments") == 1

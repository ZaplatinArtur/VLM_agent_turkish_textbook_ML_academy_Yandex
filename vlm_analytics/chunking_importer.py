from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import Database


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def import_chunking_experiment(
    database: Database,
    *,
    corpus_report: Path,
    localization_report: Path,
    refinement_report: Path,
    audit_report: Path,
    name: str = "Страница → смысловые блоки v3",
) -> int:
    paths = (
        corpus_report.resolve(),
        localization_report.resolve(),
        refinement_report.resolve(),
        audit_report.resolve(),
    )
    payloads = [_read(path) for path in paths]
    corpus, localization, refinement, audit = payloads
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    experiment_key = f"hybrid_chunking:{digest.hexdigest()[:24]}"
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    page_metrics = localization["page_chunking"]
    hybrid_metrics = localization["hybrid_task_chunking"]
    metrics = [
        (
            "hit_at_1",
            "Hit@1",
            "Локализация задания",
            float(page_metrics["hit_at_1"]) * 100,
            float(hybrid_metrics["hit_at_1"]) * 100,
            "%",
            int(localization["queries"]),
            10,
            "Точный/почти точный запрос по заданию с длинной страницы.",
        ),
        (
            "hit_at_5",
            "Hit@5",
            "Локализация задания",
            float(page_metrics["hit_at_5"]) * 100,
            float(hybrid_metrics["hit_at_5"]) * 100,
            "%",
            int(localization["queries"]),
            20,
            "Не является конечной accuracy агента.",
        ),
        (
            "mrr_at_5",
            "MRR@5",
            "Локализация задания",
            float(page_metrics["mrr_at_5"]) * 100,
            float(hybrid_metrics["mrr_at_5"]) * 100,
            "%",
            int(localization["queries"]),
            30,
            "Позиция целевого фрагмента в первой пятёрке.",
        ),
        (
            "chunks",
            "Количество чанков",
            "Корпус",
            float(corpus["pages"]),
            float(corpus["units"]),
            "шт.",
            int(corpus["pages"]),
            40,
            f"{corpus['books']} учебников.",
        ),
        (
            "units_per_page",
            "Блоков на страницу",
            "Корпус",
            1.0,
            float(corpus["units_per_page"]),
            "×",
            int(corpus["pages"]),
            50,
            "Среднее число смысловых блоков.",
        ),
        (
            "oversized_units",
            "Слишком длинные блоки",
            "Качество чанков",
            None,
            float(corpus["oversized_units"]),
            "шт.",
            int(corpus["units"]),
            60,
            "Требуют дополнительной нарезки или layout-данных.",
        ),
        (
            "qwen_rule_agreement",
            "Rules ↔ Qwen agreement",
            "Qwen-аудит",
            None,
            float(audit["agreement_rate"]) * 100,
            "%",
            int(audit["audited_units"]),
            70,
            "Только неоднозначные страницы; Qwen не является gold-разметкой.",
        ),
        (
            "qwen_change_rate",
            "Изменено Qwen",
            "Qwen-аудит",
            None,
            float(refinement["change_rate"]) * 100,
            "%",
            int(refinement["refined_units"]),
            80,
            "Доля изменённых типов в специально сложной выборке.",
        ),
        (
            "qwen_schema_success",
            "Корректный формат Qwen",
            "Надёжность",
            100.0
            * float(refinement["successful_pages"])
            / max(int(refinement["selected_pages"]), 1),
            100.0
            * float(audit["successful_pages"])
            / max(int(audit["sampled_pages"]), 1),
            "%",
            int(audit["sampled_pages"]),
            90,
            "До и после восстановления пропущенных блоков.",
        ),
        (
            "chunking_runtime",
            "Время нарезки корпуса",
            "Производительность",
            None,
            float(corpus["runtime_seconds"]),
            "с",
            int(corpus["pages"]),
            100,
            "Без полного Qwen-refinement.",
        ),
    ]
    metadata = {
        "reports": {path.name: payload for path, payload in zip(paths, payloads)},
        "source_paths": [str(path) for path in paths],
        "unit_kinds": corpus.get("unit_kinds", {}),
        "qwen_transitions": refinement.get("transitions", {}),
        "diagnostic": localization.get("diagnostic"),
    }

    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO retrieval_experiments(
                experiment_key, name, created_at, dataset, method, notes,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_key) DO UPDATE SET
                name = excluded.name,
                dataset = excluded.dataset,
                method = excluded.method,
                notes = excluded.notes,
                metadata_json = excluded.metadata_json
            """,
            (
                experiment_key,
                name,
                created_at,
                "42 981 страница / 200 учебников",
                "Текстовые правила + выборочный Qwen 3.5 9B",
                "Task-aware пилот. Layout/VLM-слой пока не подключён.",
                Database.json_value(metadata),
            ),
        )
        experiment_id = int(
            connection.execute(
                "SELECT id FROM retrieval_experiments WHERE experiment_key = ?",
                (experiment_key,),
            ).fetchone()[0]
        )
        connection.execute(
            "DELETE FROM retrieval_experiment_metrics WHERE experiment_id = ?",
            (experiment_id,),
        )
        connection.executemany(
            """
            INSERT INTO retrieval_experiment_metrics(
                experiment_id, metric_key, label, category, baseline_value,
                candidate_value, unit, sample_size, sort_order, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(experiment_id, *metric) for metric in metrics],
        )
    return experiment_id

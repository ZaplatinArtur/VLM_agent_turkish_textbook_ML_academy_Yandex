from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .database import Database


AUDIT_NAME = "Ручной аудит judge v8 — ответ 0/1 + отдельный reasoning"
METHODOLOGY = (
    "25 стратифицированных заданий × 4 режима. Финальный ответ размечен "
    "отдельно как 0/1, корректность рассуждения — отдельным бинарным полем. "
    "Качество reasoning не меняет score финального ответа. Эталоны-картинки "
    "транскрибированы один раз; choice и numeric проверяются детерминированно, "
    "открытые ответы — семантически через Qwen; битые эталоны блокируются."
)


def import_manual_audit(database: Database, path: Path) -> int:
    path = Path(path)
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    records = [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Файл ручной разметки пуст.")
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with database.transaction() as connection:
        existing = connection.execute(
            "SELECT id FROM judge_audits WHERE source_fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if existing:
            return int(existing[0])
        cursor = connection.execute(
            """
            INSERT INTO judge_audits(
                name, created_at, methodology, source_fingerprint,
                sample_tasks, sample_answers, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                AUDIT_NAME,
                now,
                METHODOLOGY,
                fingerprint,
                len({row["task_id"] for row in records}),
                len(records),
                Database.json_value(
                    {
                        "source": str(path.resolve()),
                        "selection": "stratified_stress_sample",
                    }
                ),
            ),
        )
        audit_id = int(cursor.lastrowid)
        for row in records:
            task_result = connection.execute(
                """
                SELECT tr.id
                FROM task_results tr
                JOIN runs r ON r.id = tr.run_id
                WHERE tr.task_id = ? AND r.run_key = ?
                ORDER BY r.id DESC
                LIMIT 1
                """,
                (row["task_id"], row["mode"]),
            ).fetchone()
            manual_answer = row.get(
                "manual_answer_correct", row.get("manual_strict_correct")
            )
            judge_answer = row.get(
                "judge_answer_correct", row.get("judge_strict_correct")
            )
            answer_agreement = row.get(
                "answer_agreement", row.get("agreement")
            )
            manual_reasoning = row.get("manual_reasoning_correct")
            judge_reasoning = row.get("judge_reasoning_correct")
            reasoning_agreement = row.get("reasoning_agreement")
            connection.execute(
                """
                INSERT INTO manual_judge_labels(
                    audit_id, task_result_id, task_id, run_key, subject,
                    answer_type, reference_kind,
                    manual_answer_correct, judge_answer_correct, answer_agreement,
                    manual_reasoning_correct, judge_reasoning_correct,
                    reasoning_agreement,
                    manual_strict_correct, judge_strict_correct, agreement,
                    manual_reference_quality_issue,
                    judge_reference_quality_issue, error_category,
                    manual_note, reference_note, reviewer, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    int(task_result[0]) if task_result else None,
                    row["task_id"],
                    row["mode"],
                    row.get("subject"),
                    row.get("answer_type"),
                    row.get("reference_kind"),
                    None if manual_answer is None else int(bool(manual_answer)),
                    None if judge_answer is None else int(bool(judge_answer)),
                    (
                        None
                        if answer_agreement is None
                        else int(bool(answer_agreement))
                    ),
                    (
                        None
                        if manual_reasoning is None
                        else int(bool(manual_reasoning))
                    ),
                    (
                        None
                        if judge_reasoning is None
                        else int(bool(judge_reasoning))
                    ),
                    (
                        None
                        if reasoning_agreement is None
                        else int(bool(reasoning_agreement))
                    ),
                    None if manual_answer is None else int(bool(manual_answer)),
                    None if judge_answer is None else int(bool(judge_answer)),
                    (
                        None
                        if answer_agreement is None
                        else int(bool(answer_agreement))
                    ),
                    int(bool(row.get("manual_reference_quality_issue"))),
                    int(bool(row.get("judge_reference_quality_issue"))),
                    row.get("error_category"),
                    row.get("manual_note"),
                    row.get("reference_note"),
                    row.get("reviewer"),
                    row.get("confidence"),
                ),
            )
    return audit_id

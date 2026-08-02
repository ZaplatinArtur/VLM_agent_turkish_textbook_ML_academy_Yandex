from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    model TEXT,
    prompt_version TEXT,
    imported_at TEXT NOT NULL,
    source_observed_at TEXT,
    raw_source TEXT,
    judge_source TEXT,
    source_fingerprint TEXT NOT NULL UNIQUE,
    record_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_key_time
ON runs(run_key, imported_at DESC);

CREATE TABLE IF NOT EXISTS task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    subject TEXT,
    grade TEXT,
    answer_type TEXT,
    reference_kind TEXT,
    question_image_url TEXT,
    reference_image_url TEXT,
    final_answer TEXT,
    solution_steps TEXT,
    reasoning TEXT,
    forced_answer INTEGER NOT NULL DEFAULT 0,
    raw_response TEXT,
    generation_json TEXT NOT NULL DEFAULT '{}',
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_s REAL,
    agent_error TEXT,
    strict_correct INTEGER,
    final_answer_correct INTEGER,
    reasoning_correct INTEGER,
    complete INTEGER,
    judge_label TEXT,
    judge_score REAL,
    judge_confidence REAL,
    judge_rationale TEXT,
    judge_error TEXT,
    reference_quality_issue INTEGER NOT NULL DEFAULT 0,
    error_types_json TEXT NOT NULL DEFAULT '[]',
    deterministic_applicable INTEGER,
    deterministic_matched INTEGER,
    deterministic_method TEXT,
    normalized_reference TEXT,
    normalized_candidate TEXT,
    raw_json TEXT NOT NULL,
    judge_json TEXT,
    UNIQUE(run_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run_subject
ON task_results(run_id, subject);
CREATE INDEX IF NOT EXISTS idx_results_task
ON task_results(task_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_result_id INTEGER NOT NULL REFERENCES task_results(id) ON DELETE CASCADE,
    call_index INTEGER NOT NULL,
    tool TEXT,
    query TEXT,
    args_json TEXT NOT NULL DEFAULT '{}',
    result_preview TEXT,
    returned_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
    returned_count INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_tools_result
ON tool_calls(task_result_id);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_key TEXT NOT NULL UNIQUE,
    component TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Новый',
    owner TEXT,
    notes TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    is_manual INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS issue_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    task_result_id INTEGER REFERENCES task_results(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    details TEXT,
    UNIQUE(issue_id, run_id, task_result_id)
);

CREATE INDEX IF NOT EXISTS idx_issue_status
ON issues(status, component, severity);

CREATE TABLE IF NOT EXISTS judge_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    methodology TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL UNIQUE,
    sample_tasks INTEGER NOT NULL,
    sample_answers INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS manual_judge_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id INTEGER NOT NULL REFERENCES judge_audits(id) ON DELETE CASCADE,
    task_result_id INTEGER REFERENCES task_results(id) ON DELETE SET NULL,
    task_id TEXT NOT NULL,
    run_key TEXT NOT NULL,
    subject TEXT,
    answer_type TEXT,
    reference_kind TEXT,
    manual_answer_correct INTEGER,
    judge_answer_correct INTEGER,
    answer_agreement INTEGER,
    manual_reasoning_correct INTEGER,
    judge_reasoning_correct INTEGER,
    reasoning_agreement INTEGER,
    manual_strict_correct INTEGER,
    judge_strict_correct INTEGER,
    agreement INTEGER,
    manual_reference_quality_issue INTEGER NOT NULL DEFAULT 0,
    judge_reference_quality_issue INTEGER NOT NULL DEFAULT 0,
    error_category TEXT,
    manual_note TEXT,
    reference_note TEXT,
    reviewer TEXT,
    confidence TEXT,
    UNIQUE(audit_id, task_id, run_key)
);

CREATE INDEX IF NOT EXISTS idx_manual_labels_audit
ON manual_judge_labels(audit_id, agreement, run_key);

CREATE TABLE IF NOT EXISTS retrieval_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    method TEXT NOT NULL,
    notes TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS retrieval_experiment_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL
        REFERENCES retrieval_experiments(id) ON DELETE CASCADE,
    metric_key TEXT NOT NULL,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    baseline_value REAL,
    candidate_value REAL,
    unit TEXT NOT NULL DEFAULT '',
    sample_size INTEGER,
    sort_order INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(experiment_id, metric_key)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_experiment_metrics
ON retrieval_experiment_metrics(experiment_id, category, sort_order);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            existing_manual_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(manual_judge_labels)"
                )
            }
            for column in (
                "manual_answer_correct",
                "judge_answer_correct",
                "answer_agreement",
                "manual_reasoning_correct",
                "judge_reasoning_correct",
                "reasoning_agreement",
            ):
                if column not in existing_manual_columns:
                    connection.execute(
                        f"ALTER TABLE manual_judge_labels ADD COLUMN {column} INTEGER"
                    )
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES('version', '5')"
            )
            connection.execute(
                """
                UPDATE issues
                SET owner = CASE
                    WHEN component = 'Retrieval' THEN 'Ретрив'
                    WHEN component = 'Judge' THEN 'Джадж'
                    WHEN component = 'Data'
                        THEN 'Комбинированный: джадж + данные'
                    WHEN component = 'Integration' THEN 'Комбинированный'
                    WHEN component = 'Agent' AND (
                        issue_type IN ('high_context', 'high_latency')
                        OR EXISTS (
                            SELECT 1
                            FROM issue_occurrences io
                            JOIN tool_calls tc
                              ON tc.task_result_id = io.task_result_id
                            WHERE io.issue_id = issues.id
                              AND (
                                  tc.error IS NOT NULL
                                  OR issue_type IN ('high_context', 'high_latency')
                              )
                        )
                    ) THEN 'Агент + ретрив'
                    WHEN component = 'Agent' THEN 'Агент'
                    ELSE 'Комбинированный'
                END
                WHERE owner IS NULL OR TRIM(owner) = ''
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def scalar(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
            return row[0] if row else None

    def rows(
        self, query: str, parameters: tuple[Any, ...] = ()
    ) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(query, parameters).fetchall())

    def get_setting(self, key: str, default: str = "") -> str:
        value = self.scalar("SELECT value FROM app_settings WHERE key = ?", (key,))
        return default if value is None else str(value)

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )

    def latest_runs(self) -> list[sqlite3.Row]:
        return self.rows(
            """
            SELECT r.*
            FROM runs r
            JOIN (
                SELECT run_key, MAX(id) AS max_id
                FROM runs
                GROUP BY run_key
            ) latest ON latest.max_id = r.id
            ORDER BY CASE r.run_key
                WHEN 'b0_no_tools' THEN 0
                WHEN 'web_search' THEN 1
                WHEN 'agent_rag' THEN 2
                WHEN 'agent_rag_thinking' THEN 3
                WHEN 'agent_rag_hybrid_chunks' THEN 4
                WHEN 'agent_rag_hybrid_chunks_thinking' THEN 5
                ELSE 99 END
            """
        )

    def run_history(self) -> list[sqlite3.Row]:
        return self.rows(
            """
            SELECT r.*,
                   SUM(CASE WHEN tr.strict_correct = 1 THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN tr.judge_error IS NOT NULL THEN 1 ELSE 0 END) AS judge_errors
            FROM runs r
            LEFT JOIN task_results tr ON tr.run_id = r.id
            GROUP BY r.id
            ORDER BY r.imported_at DESC, r.id DESC
            """
        )

    @staticmethod
    def json_value(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

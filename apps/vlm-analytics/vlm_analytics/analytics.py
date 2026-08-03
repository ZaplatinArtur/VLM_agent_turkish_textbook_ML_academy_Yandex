from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import MODE_ORDER
from .database import Database


def percentage(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 1)


@dataclass(frozen=True)
class ModeSummary:
    run_id: int
    run_key: str
    display_name: str
    imported_at: str
    total: int
    correct: int
    accuracy: float
    forced: int
    forced_rate: float
    reasoning: int
    reasoning_rate: float
    search_tasks: int
    search_rate: float
    tool_calls: int
    tool_errors: int
    judge_errors: int
    reference_issues: int
    final_strict_disagreements: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency_s: float


class AnalyticsService:
    def __init__(self, database: Database):
        self.database = database

    def latest_run_ids(self) -> list[int]:
        return [int(row["id"]) for row in self.database.latest_runs()]

    def mode_summaries(self) -> list[ModeSummary]:
        summaries: list[ModeSummary] = []
        for run in self.database.latest_runs():
            run_id = int(run["id"])
            row = self.database.rows(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN strict_correct = 1 THEN 1 ELSE 0 END) AS correct,
                    SUM(CASE WHEN forced_answer = 1 THEN 1 ELSE 0 END) AS forced,
                    SUM(CASE WHEN reasoning IS NOT NULL AND TRIM(reasoning) <> ''
                             THEN 1 ELSE 0 END) AS reasoning_count,
                    SUM(CASE WHEN judge_error IS NOT NULL THEN 1 ELSE 0 END)
                        AS judge_errors,
                    SUM(reference_quality_issue) AS reference_issues,
                    SUM(CASE WHEN final_answer_correct = 1 AND strict_correct = 0
                             THEN 1 ELSE 0 END) AS final_strict_disagreements,
                    AVG(input_tokens) AS avg_input_tokens,
                    AVG(output_tokens) AS avg_output_tokens,
                    AVG(latency_s) AS avg_latency_s
                FROM task_results
                WHERE run_id = ?
                """,
                (run_id,),
            )[0]
            tools = self.database.rows(
                """
                SELECT
                    COUNT(*) AS calls,
                    COUNT(DISTINCT task_result_id) AS search_tasks,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)
                        AS tool_errors
                FROM tool_calls
                WHERE task_result_id IN (
                    SELECT id FROM task_results WHERE run_id = ?
                )
                """,
                (run_id,),
            )[0]
            total = int(row["total"] or 0)
            correct = int(row["correct"] or 0)
            forced = int(row["forced"] or 0)
            reasoning = int(row["reasoning_count"] or 0)
            search_tasks = int(tools["search_tasks"] or 0)
            summaries.append(
                ModeSummary(
                    run_id=run_id,
                    run_key=str(run["run_key"]),
                    display_name=str(run["display_name"]),
                    imported_at=str(run["imported_at"]),
                    total=total,
                    correct=correct,
                    accuracy=percentage(correct, total),
                    forced=forced,
                    forced_rate=percentage(forced, total),
                    reasoning=reasoning,
                    reasoning_rate=percentage(reasoning, total),
                    search_tasks=search_tasks,
                    search_rate=percentage(search_tasks, total),
                    tool_calls=int(tools["calls"] or 0),
                    tool_errors=int(tools["tool_errors"] or 0),
                    judge_errors=int(row["judge_errors"] or 0),
                    reference_issues=int(row["reference_issues"] or 0),
                    final_strict_disagreements=int(
                        row["final_strict_disagreements"] or 0
                    ),
                    avg_input_tokens=round(float(row["avg_input_tokens"] or 0), 1),
                    avg_output_tokens=round(float(row["avg_output_tokens"] or 0), 1),
                    avg_latency_s=round(float(row["avg_latency_s"] or 0), 1),
                )
            )
        return sorted(
            summaries, key=lambda item: MODE_ORDER.get(item.run_key, 99)
        )

    def subject_matrix(self) -> tuple[list[str], list[ModeSummary], dict]:
        modes = self.mode_summaries()
        values: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for mode in modes:
            rows = self.database.rows(
                """
                SELECT subject, COUNT(*) AS total,
                       SUM(CASE WHEN strict_correct = 1 THEN 1 ELSE 0 END)
                           AS correct
                FROM task_results
                WHERE run_id = ?
                GROUP BY subject
                ORDER BY subject
                """,
                (mode.run_id,),
            )
            for row in rows:
                total = int(row["total"] or 0)
                correct = int(row["correct"] or 0)
                values[str(row["subject"])][mode.run_key] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": percentage(correct, total),
                }
        subjects = sorted(values, key=lambda value: value.casefold())
        return subjects, modes, values

    def run_metrics(
        self,
        metric: str,
        *,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        metric_sql = {
            "accuracy": (
                "100.0 * SUM(CASE WHEN tr.strict_correct = 1 THEN 1 ELSE 0 END)"
                " / NULLIF(COUNT(*), 0)"
            ),
            "forced_rate": (
                "100.0 * SUM(CASE WHEN tr.forced_answer = 1 THEN 1 ELSE 0 END)"
                " / NULLIF(COUNT(*), 0)"
            ),
            "reasoning_rate": (
                "100.0 * SUM(CASE WHEN tr.reasoning IS NOT NULL "
                "AND TRIM(tr.reasoning) <> '' THEN 1 ELSE 0 END)"
                " / NULLIF(COUNT(*), 0)"
            ),
            "avg_latency": "AVG(tr.latency_s)",
            "avg_input_tokens": "AVG(tr.input_tokens)",
            "avg_output_tokens": "AVG(tr.output_tokens)",
            "judge_error_rate": (
                "100.0 * SUM(CASE WHEN tr.judge_error IS NOT NULL THEN 1 ELSE 0 END)"
                " / NULLIF(COUNT(*), 0)"
            ),
        }
        expression = metric_sql.get(metric, metric_sql["accuracy"])
        parameters: list[Any] = []
        subject_clause = ""
        if subject and subject != "Все предметы":
            subject_clause = "AND tr.subject = ?"
            parameters.append(subject)
        rows = self.database.rows(
            f"""
            SELECT r.id, r.run_key, r.display_name, r.imported_at,
                   r.source_observed_at, {expression} AS value,
                   COUNT(*) AS total
            FROM runs r
            JOIN task_results tr ON tr.run_id = r.id
            WHERE 1 = 1 {subject_clause}
            GROUP BY r.id
            ORDER BY r.imported_at, r.id
            """,
            tuple(parameters),
        )
        return [dict(row) for row in rows]

    def task_rows(
        self,
        *,
        run_id: int | None = None,
        subject: str | None = None,
        search: str | None = None,
        only_problems: bool = False,
    ) -> list[dict[str, Any]]:
        run_ids = [run_id] if run_id else self.latest_run_ids()
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        clauses = [f"tr.run_id IN ({placeholders})"]
        parameters: list[Any] = list(run_ids)
        if subject and subject != "Все предметы":
            clauses.append("tr.subject = ?")
            parameters.append(subject)
        if search:
            clauses.append(
                "(tr.task_id LIKE ? OR tr.final_answer LIKE ? OR tr.subject LIKE ?)"
            )
            value = f"%{search}%"
            parameters.extend((value, value, value))
        if only_problems:
            clauses.append(
                "(tr.strict_correct = 0 OR tr.agent_error IS NOT NULL "
                "OR tr.judge_error IS NOT NULL OR tr.forced_answer = 1)"
            )
        rows = self.database.rows(
            f"""
            SELECT tr.*, r.display_name, r.run_key, r.imported_at,
                   (SELECT COUNT(*) FROM tool_calls tc
                    WHERE tc.task_result_id = tr.id) AS tool_call_count
            FROM task_results tr
            JOIN runs r ON r.id = tr.run_id
            WHERE {' AND '.join(clauses)}
            ORDER BY tr.task_id, r.id DESC
            """,
            tuple(parameters),
        )
        return [dict(row) for row in rows]

    def tool_stats(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mode in self.mode_summaries():
            calls = self.database.rows(
                """
                SELECT tc.*
                FROM tool_calls tc
                JOIN task_results tr ON tr.id = tc.task_result_id
                WHERE tr.run_id = ?
                """,
                (mode.run_id,),
            )
            no_result = sum(
                1
                for call in calls
                if "bulunamad" in str(call["result_preview"] or "").casefold()
                or "no result" in str(call["result_preview"] or "").casefold()
                or "not found" in str(call["result_preview"] or "").casefold()
            )
            returned = sum(int(call["returned_count"] or 0) for call in calls)
            rows.append(
                {
                    "run_key": mode.run_key,
                    "display_name": mode.display_name,
                    "tasks": mode.total,
                    "search_tasks": mode.search_tasks,
                    "search_rate": mode.search_rate,
                    "calls": len(calls),
                    "unique_queries": len(
                        {str(call["query"]) for call in calls if call["query"]}
                    ),
                    "errors": sum(1 for call in calls if call["error"]),
                    "no_result": no_result,
                    "returned_chunks": returned,
                    "accuracy": mode.accuracy,
                }
            )
        return rows

    def latest_retrieval_experiment(self) -> dict[str, Any] | None:
        experiments = self.database.rows(
            "SELECT * FROM retrieval_experiments ORDER BY id DESC LIMIT 1"
        )
        if not experiments:
            return None
        experiment = dict(experiments[0])
        experiment["metadata"] = self.decode_json(
            experiment.pop("metadata_json", "{}"), {}
        )
        experiment["metrics"] = [
            dict(row)
            for row in self.database.rows(
                """
                SELECT *
                FROM retrieval_experiment_metrics
                WHERE experiment_id = ?
                ORDER BY sort_order, id
                """,
                (int(experiment["id"]),),
            )
        ]
        return experiment

    def retrieval_experiment_history(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.database.rows(
                """
                SELECT re.id, re.name, re.created_at, re.dataset, re.method,
                       MAX(CASE WHEN rem.metric_key = 'hit_at_1'
                                THEN rem.candidate_value END) AS hit_at_1,
                       MAX(CASE WHEN rem.metric_key = 'hit_at_5'
                                THEN rem.candidate_value END) AS hit_at_5,
                       MAX(CASE WHEN rem.metric_key = 'mrr_at_5'
                                THEN rem.candidate_value END) AS mrr_at_5
                FROM retrieval_experiments re
                LEFT JOIN retrieval_experiment_metrics rem
                  ON rem.experiment_id = re.id
                GROUP BY re.id
                ORDER BY re.created_at, re.id
                """
            )
        ]

    def judge_stats(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mode in self.mode_summaries():
            data = self.database.rows(
                """
                SELECT judge_label, COUNT(*) AS count
                FROM task_results
                WHERE run_id = ?
                GROUP BY judge_label
                """,
                (mode.run_id,),
            )
            labels = {
                str(row["judge_label"] or "evaluation_failure"): int(row["count"])
                for row in data
            }
            rows.append(
                {
                    "run_key": mode.run_key,
                    "display_name": mode.display_name,
                    "labels": labels,
                    "judge_errors": mode.judge_errors,
                    "reference_issues": mode.reference_issues,
                    "final_strict_disagreements": mode.final_strict_disagreements,
                }
            )
        return rows

    def issue_rows(
        self,
        *,
        component: str | None = None,
        owner: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["1 = 1"]
        parameters: list[Any] = []
        if component and component != "Все":
            clauses.append("i.component = ?")
            parameters.append(component)
        if owner and owner != "Все":
            if owner == "Комбинированные":
                clauses.append(
                    "(i.owner LIKE 'Комбинированный%' OR i.owner LIKE '% + %')"
                )
            else:
                clauses.append("i.owner = ?")
                parameters.append(owner)
        if status and status != "Все":
            clauses.append("i.status = ?")
            parameters.append(status)
        rows = self.database.rows(
            f"""
            SELECT i.*,
                   (SELECT io.task_result_id
                    FROM issue_occurrences io
                    WHERE io.issue_id = i.id
                    ORDER BY io.id DESC LIMIT 1) AS latest_task_result_id,
                   (SELECT tr.task_id
                    FROM issue_occurrences io
                    JOIN task_results tr ON tr.id = io.task_result_id
                    WHERE io.issue_id = i.id
                    ORDER BY io.id DESC LIMIT 1) AS task_id,
                   (SELECT r.display_name
                    FROM issue_occurrences io
                    JOIN runs r ON r.id = io.run_id
                    WHERE io.issue_id = i.id
                    ORDER BY io.id DESC LIMIT 1) AS latest_run
            FROM issues i
            WHERE {' AND '.join(clauses)}
            ORDER BY CASE i.severity
                WHEN 'Критический' THEN 0
                WHEN 'Высокий' THEN 1
                WHEN 'Средний' THEN 2
                ELSE 3 END,
                i.last_seen_at DESC
            """,
            tuple(parameters),
        )
        return [dict(row) for row in rows]

    def update_issue(
        self,
        issue_id: int,
        *,
        status: str,
        owner: str,
        notes: str,
        severity: str,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE issues
                SET status = ?, owner = ?, notes = ?, severity = ?
                WHERE id = ?
                """,
                (status, owner, notes, severity, issue_id),
            )

    def add_manual_issue(
        self,
        *,
        component: str,
        title: str,
        severity: str,
        owner: str,
        notes: str,
    ) -> None:
        import hashlib
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        key = hashlib.sha256(
            f"manual:{component}:{title}:{now}".encode("utf-8")
        ).hexdigest()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO issues(
                    issue_key, component, issue_type, severity, title, status,
                    owner, notes, first_seen_at, last_seen_at,
                    occurrence_count, is_manual
                ) VALUES (?, ?, 'manual', ?, ?, 'Новый', ?, ?, ?, ?, 1, 1)
                """,
                (key, component, severity, title, owner, notes, now, now),
            )

    def issue_breakdown(self) -> dict[str, int]:
        return {
            str(row["component"]): int(row["count"])
            for row in self.database.rows(
                """
                SELECT component, COUNT(*) AS count
                FROM issues
                WHERE status <> 'Исправлен'
                GROUP BY component
                """
            )
        }

    def latest_judge_audit(self) -> dict[str, Any] | None:
        audit = self.database.rows(
            "SELECT * FROM judge_audits ORDER BY id DESC LIMIT 1"
        )
        if not audit:
            return None
        result = dict(audit[0])
        audit_id = int(result["id"])
        counts = dict(
            self.database.rows(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) IS NOT NULL
                             AND COALESCE(judge_answer_correct, judge_strict_correct) IS NOT NULL
                             THEN 1 ELSE 0 END) AS evaluable,
                    SUM(CASE WHEN COALESCE(answer_agreement, agreement) = 1
                             THEN 1 ELSE 0 END) AS agreed,
                    SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 1
                              AND COALESCE(judge_answer_correct, judge_strict_correct) = 1
                             THEN 1 ELSE 0 END) AS tp,
                    SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 0
                              AND COALESCE(judge_answer_correct, judge_strict_correct) = 0
                             THEN 1 ELSE 0 END) AS tn,
                    SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 0
                              AND COALESCE(judge_answer_correct, judge_strict_correct) = 1
                             THEN 1 ELSE 0 END) AS fp,
                    SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 1
                              AND COALESCE(judge_answer_correct, judge_strict_correct) = 0
                             THEN 1 ELSE 0 END) AS fn,
                    SUM(CASE WHEN manual_reasoning_correct IS NOT NULL
                              AND judge_reasoning_correct IS NOT NULL
                             THEN 1 ELSE 0 END) AS reasoning_evaluable,
                    SUM(CASE WHEN reasoning_agreement = 1
                             THEN 1 ELSE 0 END) AS reasoning_agreed,
                    SUM(CASE WHEN manual_reference_quality_issue = 1
                             THEN 1 ELSE 0 END) AS ref_positive,
                    SUM(CASE WHEN manual_reference_quality_issue = 1
                              AND judge_reference_quality_issue = 1
                             THEN 1 ELSE 0 END) AS ref_detected,
                    SUM(CASE WHEN manual_reference_quality_issue = 0
                              AND judge_reference_quality_issue = 1
                             THEN 1 ELSE 0 END) AS ref_false_positive
                FROM manual_judge_labels
                WHERE audit_id = ?
                """,
                (audit_id,),
            )[0]
        )
        for key, value in counts.items():
            result[key] = int(value or 0)
        evaluable = result["evaluable"]
        result["accuracy"] = percentage(result["agreed"], evaluable)
        result["precision"] = percentage(
            result["tp"], result["tp"] + result["fp"]
        )
        result["recall"] = percentage(
            result["tp"], result["tp"] + result["fn"]
        )
        result["specificity"] = percentage(
            result["tn"], result["tn"] + result["fp"]
        )
        result["reasoning_accuracy"] = percentage(
            result["reasoning_agreed"], result["reasoning_evaluable"]
        )
        manual_positive = (result["tp"] + result["fn"]) / max(evaluable, 1)
        judge_positive = (result["tp"] + result["fp"]) / max(evaluable, 1)
        expected = manual_positive * judge_positive + (
            1 - manual_positive
        ) * (1 - judge_positive)
        observed = result["agreed"] / max(evaluable, 1)
        result["kappa"] = (
            (observed - expected) / (1 - expected)
            if expected < 1
            else 1.0
        )
        result["reference_recall"] = percentage(
            result["ref_detected"], result["ref_positive"]
        )
        return result

    def judge_audit_breakdown(self, field: str = "run_key") -> list[dict[str, Any]]:
        allowed = {
            "run_key": "run_key",
            "subject": "subject",
            "answer_type": "answer_type",
        }
        column = allowed.get(field, "run_key")
        audit = self.latest_judge_audit()
        if not audit:
            return []
        rows = self.database.rows(
            f"""
            SELECT {column} AS label,
                   COUNT(*) AS total,
                   SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) IS NOT NULL
                            THEN 1 ELSE 0 END) AS evaluable,
                   SUM(CASE WHEN COALESCE(answer_agreement, agreement) = 1
                            THEN 1 ELSE 0 END) AS agreed,
                   SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 0
                             AND COALESCE(judge_answer_correct, judge_strict_correct) = 1
                            THEN 1 ELSE 0 END) AS fp,
                   SUM(CASE WHEN COALESCE(manual_answer_correct, manual_strict_correct) = 1
                             AND COALESCE(judge_answer_correct, judge_strict_correct) = 0
                            THEN 1 ELSE 0 END) AS fn
            FROM manual_judge_labels
            WHERE audit_id = ?
            GROUP BY {column}
            ORDER BY CASE {column}
                WHEN 'b0_no_tools' THEN 0
                WHEN 'web_search' THEN 1
                WHEN 'agent_rag' THEN 2
                WHEN 'agent_rag_thinking' THEN 3
                ELSE 99 END,
                {column}
            """,
            (int(audit["id"]),),
        )
        result = []
        for row in rows:
            item = dict(row)
            item["accuracy"] = percentage(
                int(item["agreed"] or 0), int(item["evaluable"] or 0)
            )
            result.append(item)
        return result

    def judge_audit_errors(self) -> list[dict[str, Any]]:
        audit = self.latest_judge_audit()
        if not audit:
            return []
        return [
            dict(row)
            for row in self.database.rows(
                """
                SELECT ml.*, r.display_name, tr.final_answer,
                       tr.judge_rationale
                FROM manual_judge_labels ml
                LEFT JOIN task_results tr ON tr.id = ml.task_result_id
                LEFT JOIN runs r ON r.id = tr.run_id
                WHERE ml.audit_id = ?
                  AND COALESCE(ml.answer_agreement, ml.agreement) = 0
                ORDER BY ml.error_category, ml.run_key, ml.task_id
                """,
                (int(audit["id"]),),
            )
        ]

    @staticmethod
    def decode_json(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

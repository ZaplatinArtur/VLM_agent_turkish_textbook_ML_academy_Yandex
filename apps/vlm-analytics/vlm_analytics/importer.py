from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DATASET_VERSION
from .database import Database


@dataclass(frozen=True)
class ImportResult:
    run_id: int
    run_key: str
    display_name: str
    records: int
    imported: bool
    message: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected JSON object")
            records.append(value)
    return records


def _fingerprint(paths: Iterable[Path], run_key: str) -> str:
    digest = hashlib.sha256(run_key.encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _reference_kind(manifest: dict[str, Any]) -> str:
    if _text(manifest.get("reference_answer")):
        return "text"
    if _text(manifest.get("reference_answer_image")):
        return "image"
    return "unknown"


def _issue_specs(
    task_id: str,
    raw: dict[str, Any],
    judge: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
) -> list[tuple[str, str, str, str, str, str]]:
    issues: list[tuple[str, str, str, str, str, str]] = []

    def add(
        component: str,
        issue_type: str,
        severity: str,
        title: str,
        details: str,
    ) -> None:
        has_tool_error = any(_text(call.get("error")) for call in tool_calls)
        if component == "Agent":
            owner = (
                "Агент + ретрив"
                if has_tool_error
                or (tool_calls and issue_type in {"high_context", "high_latency"})
                else "Агент"
            )
        elif component == "Retrieval":
            owner = "Ретрив"
        elif component == "Judge":
            owner = "Джадж"
        elif component == "Data":
            owner = "Комбинированный: джадж + данные"
        else:
            owner = "Комбинированный"
        issues.append((component, issue_type, severity, title, details, owner))

    if raw.get("error"):
        add("Agent", "agent_error", "Критический", "Ошибка агента", str(raw["error"]))
    if raw.get("forced_answer"):
        add(
            "Agent",
            "forced_answer",
            "Средний",
            "Принудительный финальный ответ",
            "Агент не завершил обычный структурированный ответ.",
        )
    if not _text(raw.get("reasoning")):
        add(
            "Agent",
            "missing_reasoning",
            "Низкий",
            "Отсутствует явный reasoning",
            "Поле reasoning пустое.",
        )
    input_tokens = (raw.get("usage") or {}).get("input_tokens")
    if isinstance(input_tokens, (int, float)) and input_tokens > 12_000:
        add(
            "Agent",
            "high_context",
            "Средний",
            "Слишком большой контекст",
            f"Входных токенов: {int(input_tokens)}.",
        )
    latency = (raw.get("usage") or {}).get("latency_s")
    if isinstance(latency, (int, float)) and latency > 120:
        add(
            "Agent",
            "high_latency",
            "Средний",
            "Высокая задержка",
            f"Время решения: {float(latency):.1f} с.",
        )

    for call in tool_calls:
        call_error = _text(call.get("error"))
        if call_error:
            add(
                "Retrieval",
                "tool_error",
                "Средний",
                "Ошибка вызова инструмента",
                call_error,
            )
        preview = str(call.get("result_preview") or "").casefold()
        if call.get("tool") == "web_search" and (
            "bulunamad" in preview or "no result" in preview or "not found" in preview
        ):
            add(
                "Retrieval",
                "web_no_results",
                "Низкий",
                "Веб-поиск не вернул результатов",
                _text(call.get("result_preview")) or "Пустой результат.",
            )

    if judge:
        judge_block = judge.get("judge") or {}
        verdict = judge.get("verdict") or {}
        if judge_block.get("error"):
            add(
                "Judge",
                "judge_error",
                "Критический",
                "Ошибка джаджа",
                str(judge_block["error"]),
            )
        if verdict.get("reference_quality_issue"):
            add(
                "Data",
                "reference_quality",
                "Высокий",
                "Проблема качества эталона",
                _text(verdict.get("rationale")) or "Джадж отметил эталон.",
            )
        if (
            verdict.get("final_answer_correct") is True
            and verdict.get("strict_correct") is False
        ):
            add(
                "Judge",
                "final_strict_disagreement",
                "Средний",
                "Финальный ответ верный, strict score = 0",
                _text(verdict.get("rationale")) or "Расхождение итоговой оценки.",
            )
        deterministic = judge.get("deterministic") or {}
        if (
            deterministic.get("applicable")
            and verdict.get("strict_correct") is not None
            and bool(deterministic.get("matched"))
            != bool(verdict.get("strict_correct"))
        ):
            add(
                "Judge",
                "deterministic_disagreement",
                "Средний",
                "LLM и детерминированная проверка не согласны",
                f"deterministic={deterministic.get('matched')}, "
                f"strict={verdict.get('strict_correct')}",
            )

    return issues


def import_run(
    database: Database,
    *,
    run_key: str,
    display_name: str,
    raw_path: Path,
    judge_path: Path,
    manifest_path: Path,
    raw_source: str | None = None,
    judge_source: str | None = None,
    dataset_version: str = DATASET_VERSION,
    observed_at: str | None = None,
) -> ImportResult:
    raw_path = Path(raw_path)
    judge_path = Path(judge_path)
    manifest_path = Path(manifest_path)
    fingerprint = _fingerprint((raw_path, judge_path, manifest_path), run_key)
    existing = database.scalar(
        "SELECT id FROM runs WHERE source_fingerprint = ?", (fingerprint,)
    )
    if existing is not None:
        return ImportResult(
            int(existing),
            run_key,
            display_name,
            int(
                database.scalar(
                    "SELECT record_count FROM runs WHERE id = ?", (existing,)
                )
                or 0
            ),
            False,
            "Такой снимок уже есть в базе.",
        )

    raw_records = _read_jsonl(raw_path)
    judge_records = _read_jsonl(judge_path)
    manifest_records = _read_jsonl(manifest_path)
    manifest = {str(row.get("task_id")): row for row in manifest_records}
    judged = {str(row.get("task_id")): row for row in judge_records}
    if not raw_records:
        raise ValueError(f"В {raw_path} нет записей")

    raw_ids = [str(row.get("task_id") or "") for row in raw_records]
    if not all(raw_ids):
        raise ValueError("В результатах есть строка без task_id")
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("В результатах обнаружены повторяющиеся task_id")

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    first = raw_records[0]
    model = _text(first.get("model"))
    prompt_version = _text(first.get("prompt_version"))
    metadata = {
        "raw_records": len(raw_records),
        "judge_records": len(judge_records),
        "manifest_records": len(manifest_records),
    }

    with database.transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs(
                run_key, display_name, dataset_version, model, prompt_version,
                imported_at, source_observed_at, raw_source, judge_source,
                source_fingerprint, record_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_key,
                display_name,
                dataset_version,
                model,
                prompt_version,
                now,
                observed_at,
                raw_source or str(raw_path),
                judge_source or str(judge_path),
                fingerprint,
                len(raw_records),
                Database.json_value(metadata),
            ),
        )
        run_id = int(cursor.lastrowid)

        for raw in raw_records:
            task_id = str(raw["task_id"])
            source = manifest.get(task_id, {})
            judge = judged.get(task_id)
            verdict = (judge or {}).get("verdict") or {}
            judge_block = (judge or {}).get("judge") or {}
            deterministic = (judge or {}).get("deterministic") or {}
            usage = raw.get("usage") or {}
            generation = raw.get("generation") or {}
            image_evidence = raw.get("image_evidence")
            if not isinstance(image_evidence, list):
                image_evidence = []
            tool_calls = raw.get("tool_calls")
            if not isinstance(tool_calls, list):
                tool_calls = []
            source_meta = source.get("source") or {}
            result_cursor = connection.execute(
                """
                INSERT INTO task_results(
                    run_id, task_id, subject, grade, answer_type, reference_kind,
                    question_image_url, reference_image_url, final_answer,
                    solution_steps, reasoning, forced_answer, raw_response,
                    exit_reason, image_evidence_json, retrieval_relevance,
                    retrieval_conflict, answer_source, experiment_id,
                    retrieval_route, retrieval_route_reason, generation_json,
                    input_tokens, output_tokens, latency_s,
                    agent_error, strict_correct, final_answer_correct,
                    reasoning_correct, complete, judge_label, judge_score,
                    judge_confidence, judge_rationale, judge_error,
                    reference_quality_issue, error_types_json,
                    deterministic_applicable, deterministic_matched,
                    deterministic_method, normalized_reference,
                    normalized_candidate, raw_json, judge_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    task_id,
                    _text((judge or {}).get("subject"))
                    or _text(source.get("subject"))
                    or "unknown",
                    _text((judge or {}).get("grade"))
                    or _text(source.get("grade")),
                    _text((judge or {}).get("answer_type"))
                    or _text(source.get("answer_type")),
                    _reference_kind(source),
                    _text(source.get("question_image")),
                    _text(source.get("reference_answer_image")),
                    _text(raw.get("final_answer")),
                    _text(raw.get("solution_steps")),
                    _text(raw.get("reasoning")),
                    int(bool(raw.get("forced_answer"))),
                    _text(raw.get("raw_response")),
                    _text(raw.get("exit_reason")),
                    Database.json_value(image_evidence),
                    _text(raw.get("retrieval_relevance")),
                    _bool(raw.get("retrieval_conflict")),
                    _text(raw.get("answer_source")),
                    _text(generation.get("experiment_id")),
                    _text(generation.get("retrieval_route")),
                    _text(generation.get("retrieval_route_reason")),
                    Database.json_value(generation),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("latency_s"),
                    _text(raw.get("error")),
                    _bool(verdict.get("strict_correct")),
                    _bool(verdict.get("final_answer_correct")),
                    _bool(verdict.get("reasoning_correct")),
                    _bool(verdict.get("complete")),
                    _text(verdict.get("label")),
                    verdict.get("score"),
                    verdict.get("confidence"),
                    _text(verdict.get("rationale")),
                    _text(judge_block.get("error")),
                    int(bool(verdict.get("reference_quality_issue"))),
                    Database.json_value(verdict.get("error_types") or []),
                    _bool(deterministic.get("applicable")),
                    _bool(deterministic.get("matched")),
                    _text(deterministic.get("method")),
                    _text(deterministic.get("normalized_reference")),
                    _text(deterministic.get("normalized_candidate")),
                    Database.json_value(raw),
                    Database.json_value(judge) if judge else None,
                ),
            )
            task_result_id = int(result_cursor.lastrowid)
            for call_index, call in enumerate(tool_calls):
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                chunk_ids = call.get("returned_chunk_ids")
                if not isinstance(chunk_ids, list):
                    chunk_ids = []
                relevance = call.get("relevance")
                if not isinstance(relevance, dict):
                    relevance = {}
                connection.execute(
                    """
                    INSERT INTO tool_calls(
                        task_result_id, call_index, tool, query, args_json,
                        result_preview, returned_chunk_ids_json, returned_count,
                        latency_ms, relevance_json, relevance_label, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_result_id,
                        call_index,
                        _text(call.get("tool")),
                        _text(args.get("query")),
                        Database.json_value(args),
                        _text(call.get("result_preview")),
                        Database.json_value(chunk_ids),
                        len(chunk_ids),
                        call.get("latency_ms"),
                        Database.json_value(relevance),
                        _text(relevance.get("label")),
                        _text(call.get("error")),
                    ),
                )

            for component, issue_type, severity, title, details, owner in _issue_specs(
                task_id, raw, judge, tool_calls
            ):
                issue_key = f"{component}:{issue_type}:{task_id}"
                connection.execute(
                    """
                    INSERT INTO issues(
                        issue_key, component, issue_type, severity, title, owner,
                        first_seen_at, last_seen_at, occurrence_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(issue_key) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        occurrence_count = issues.occurrence_count + 1,
                        severity = excluded.severity,
                        title = excluded.title,
                        owner = COALESCE(NULLIF(issues.owner, ''), excluded.owner)
                    """,
                    (
                        issue_key,
                        component,
                        issue_type,
                        severity,
                        title,
                        owner,
                        now,
                        now,
                    ),
                )
                issue_id = connection.execute(
                    "SELECT id FROM issues WHERE issue_key = ?", (issue_key,)
                ).fetchone()[0]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO issue_occurrences(
                        issue_id, run_id, task_result_id, observed_at, details
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (issue_id, run_id, task_result_id, now, details),
                )

    return ImportResult(
        run_id,
        run_key,
        display_name,
        len(raw_records),
        True,
        f"Импортировано {len(raw_records)} задач.",
    )

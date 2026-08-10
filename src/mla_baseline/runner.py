"""Батч-прогон: tasks.jsonl × solver -> results/<condition>_<prompt_version>.jsonl.

Особенности:
- resume: task_id, уже присутствующие в выходном файле, пропускаются;
- конкурентность: ограниченный пул запросов к OpenAI-совместимому backend;
- --dry-run: собирает сообщения без удалённого вызова модели.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import SecretStr

from .config import get_settings
from .contracts import Task
from .solvers import SOLVERS


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(Task.model_validate_json(line))
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: невалидная задача: {exc}") from exc
    return tasks


def load_done_ids(out_path: Path, *, retry_errors: bool = False) -> set[str]:
    if not out_path.exists():
        return set()
    records: list[dict] = []
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not retry_errors:
        return {str(record["task_id"]) for record in records}

    latest = {str(record["task_id"]): record for record in records}
    successful = [record for record in latest.values() if not record.get("error")]
    temporary = out_path.with_suffix(out_path.suffix + ".retry.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for record in successful:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(out_path)
    return {str(record["task_id"]) for record in successful}


def describe_messages(task: Task, messages: list) -> str:
    human = messages[-1]
    n_images = sum(1 for b in human.content if b.get("type") == "image_url")
    texts = [b["text"] for b in human.content if b.get("type") == "text"]
    return (
        f"{task.task_id}: {n_images} img, text blocks: "
        + " | ".join(t[:60].replace("\n", " ") for t in texts)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогон бейзлайна по набору задач")
    parser.add_argument("--tasks", type=Path, required=True, help="JSONL с Task")
    parser.add_argument("--condition", choices=sorted(SOLVERS), default="b0_no_tools")
    parser.add_argument("--limit", type=int, default=None, help="только первые N задач")
    parser.add_argument("--out", type=Path, default=None, help="путь к результату")
    parser.add_argument("--dry-run", action="store_true", help="собрать вход без вызова модели")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="сохранить успешные строки и повторить только записи с error",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="после прогона сгенерировать HTML-отчёт рядом с результатом",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="meta JSONL для срезов и ответов-картинок в отчёте",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if (
        args.dry_run
        and settings.llm_provider == "openrouter"
        and settings.openrouter_api_key is None
    ):
        settings = settings.model_copy(
            update={"openrouter_api_key": SecretStr("dry-run-not-sent")}
        )
    solver = SOLVERS[args.condition](settings)

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    if args.dry_run:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        for task in tasks:
            print(describe_messages(task, solver.build_messages(task)))
        print(f"[dry-run] OK: {len(tasks)} задач, condition={args.condition}, "
              f"model={settings.llm_model_name}, prompt={settings.prompt_version}")
        return 0

    out_path = args.out or (
        settings.results_dir / f"{args.condition}_{settings.prompt_version}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(out_path, retry_errors=args.retry_errors)
    todo = [t for t in tasks if t.task_id not in done]
    print(f"Задач: {len(tasks)}, уже готово: {len(tasks) - len(todo)}, к прогону: {len(todo)}")

    write_lock = threading.Lock()
    errors = 0
    with out_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(solver.solve, task): task for task in todo}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result.error:
                    errors += 1
                with write_lock:
                    out_fh.write(result.model_dump_json() + "\n")
                    out_fh.flush()
                status = result.error or "ok"
                print(f"[{i}/{len(todo)}] {result.task_id}: {status} "
                      f"({result.usage.latency_s}s)")

    print(f"Готово: {len(todo)} прогнано, ошибок: {errors}, результат: {out_path}")

    if args.report:
        from .report import generate

        meta = args.meta
        if meta is None:
            candidate = args.tasks.with_suffix(".meta.jsonl")
            meta = candidate if candidate.exists() else None
        report_path = generate(out_path, args.tasks, meta)
        print(f"Отчёт: {report_path}")

    from .tracing import flush

    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

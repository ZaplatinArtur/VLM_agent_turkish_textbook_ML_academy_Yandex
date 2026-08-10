"""End-to-end accuracy experiment for frozen reranker rankings.

The reranker archive already contains the same Dense candidate pool ordered by
multiple rerankers.  This module turns each top-k into an agent-visible frozen
tool response, runs the same image-first answer path, and compares the resulting
judge verdicts without invoking retrieval again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, BinaryIO

from .config import Settings, get_settings
from .context_order_experiment import FrozenContextOrderSolver
from .contracts import Task
from .runner import load_done_ids, load_tasks
from .schemas import SolveResult
from .tools.textbook_search import format_search_result_for_model

RANKING_FILES = {
    "dense": "rankings_dense.jsonl",
    "gte_multilingual": "rankings_gte_multilingual.jsonl",
    "bge_v2_m3": "rankings_bge_v2_m3.jsonl",
    "qwen3_reranker_06b": "rankings_qwen3_reranker_06b.jsonl",
}
DEFAULT_ARMS = ("dense", "gte_multilingual", "bge_v2_m3")


def _read_jsonl_stream(source: BinaryIO, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(source, 1):
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_no}: expected a JSON object")
        rows.append(value)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as source:
        return _read_jsonl_stream(source, label=str(path))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rankings_archive(
    path: Path,
    *,
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Read the selected ranking arms without extracting the archive."""

    unknown = [arm for arm in arms if arm not in RANKING_FILES]
    if unknown:
        raise ValueError(f"unknown reranker arms: {unknown}")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "experiment_manifest.json" not in names:
            raise ValueError("ranking archive has no experiment_manifest.json")
        manifest = json.loads(archive.read("experiment_manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("experiment_manifest.json must contain an object")

        rows_by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm in arms:
            filename = RANKING_FILES[arm]
            if filename not in names:
                raise ValueError(f"ranking archive has no {filename}")
            with archive.open(filename) as source:
                rows_by_arm[arm] = _read_jsonl_stream(source, label=filename)
    return manifest, rows_by_arm


def _index_rankings(
    rows: list[dict[str, Any]],
    *,
    arm: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{arm}: ranking row has no task_id")
        if task_id in indexed:
            raise ValueError(f"{arm}: duplicate task_id {task_id}")
        rankings = row.get("rankings")
        if not isinstance(rankings, list):
            raise ValueError(f"{arm}/{task_id}: rankings must be a list")
        chunk_ids: list[str] = []
        for expected_rank, hit in enumerate(rankings, 1):
            if not isinstance(hit, dict):
                raise ValueError(f"{arm}/{task_id}: ranking hit must be an object")
            chunk_id = str(hit.get("chunk_id") or "")
            if not chunk_id:
                raise ValueError(f"{arm}/{task_id}: hit has no chunk_id")
            if hit.get("rank") != expected_rank:
                raise ValueError(
                    f"{arm}/{task_id}: expected rank {expected_rank}, "
                    f"got {hit.get('rank')}"
                )
            score = hit.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"{arm}/{task_id}: score must be finite")
            chunk_ids.append(chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"{arm}/{task_id}: duplicate chunk_id in rankings")
        indexed[task_id] = row
    return indexed


def validate_ranking_arms(
    *,
    tasks: list[Task],
    rows_by_arm: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    """Ensure every arm reranks the same per-task candidate pool."""

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("tasks contain duplicate task_id values")
    expected_ids = set(task_ids)
    indexed = {
        arm: _index_rankings(rows, arm=arm) for arm, rows in rows_by_arm.items()
    }
    if "dense" not in indexed:
        raise ValueError("dense arm is required as the paired baseline")

    for arm, by_task in indexed.items():
        actual_ids = set(by_task)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)[:5]
            extra = sorted(actual_ids - expected_ids)[:5]
            raise ValueError(
                f"{arm}: task IDs differ from tasks; missing={missing}, extra={extra}"
            )

    dense = indexed["dense"]
    rerankable: set[str] = set()
    for task_id in task_ids:
        dense_row = dense[task_id]
        dense_rankings = dense_row["rankings"]
        dense_ids = [str(hit["chunk_id"]) for hit in dense_rankings]
        if dense_ids:
            rerankable.add(task_id)
        dense_pool = set(dense_ids)
        dense_query = str(dense_row.get("query") or "")
        for arm, by_task in indexed.items():
            row = by_task[task_id]
            candidate_ids = [str(hit["chunk_id"]) for hit in row["rankings"]]
            if len(candidate_ids) != len(dense_ids) or set(candidate_ids) != dense_pool:
                raise ValueError(
                    f"{arm}/{task_id}: candidate pool differs from dense"
                )
            if str(row.get("query") or "") != dense_query:
                raise ValueError(f"{arm}/{task_id}: query differs from dense")
    return indexed, rerankable


def _agent_hit(hit: dict[str, Any], *, rank: int) -> dict[str, Any]:
    metadata = {
        "subject": hit.get("subject"),
        "grade": hit.get("grade"),
        "textbook": hit.get("textbook"),
        "page": hit.get("page"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    result = {
        "chunk_id": hit.get("chunk_id"),
        "page_id": hit.get("page_id"),
        "rank": rank,
        "score": hit.get("score"),
        "subject": hit.get("subject"),
        "grade": hit.get("grade"),
        "book_id": hit.get("textbook"),
        "page_number": hit.get("page"),
        "metadata": metadata,
        "text": str(hit.get("text") or ""),
    }
    return {key: value for key, value in result.items() if value is not None}


def build_context_records(
    *,
    tasks: list[Task],
    rankings_by_task: dict[str, dict[str, Any]],
    arm: str,
    top_k: int,
    max_text_chars: int,
) -> list[dict[str, Any]]:
    """Convert saved rankings into the frozen tool-response contract."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    records: list[dict[str, Any]] = []
    for task in tasks:
        row = rankings_by_task[task.task_id]
        query = str(row.get("query") or "").strip()
        topic = (query or task.subject or "task")[:200]
        source_rankings = row["rankings"]
        hits = [
            _agent_hit(hit, rank=rank)
            for rank, hit in enumerate(source_rankings[:top_k], 1)
        ]
        relevance = {
            "label": "frozen" if hits else "empty",
            "is_useful": bool(hits),
            "top_score": hits[0].get("score") if hits else None,
            "reason": (
                f"precomputed {arm} top-{top_k}"
                if hits
                else "precomputed candidate pool is empty"
            ),
        }
        payload = {
            "query": query,
            "top_k": top_k,
            "context_order": "score",
            "mode": "or",
            "filters": {
                "subject": row.get("subject"),
                "grade": row.get("grade"),
            },
            "retrieved": len(source_rankings),
            "returned": len(hits),
            "relevance": relevance,
            "hits": hits,
            "frozen": True,
            "reranker_arm": arm,
        }
        compact_payload = json.loads(
            format_search_result_for_model(
                payload,
                max_text_chars=max_text_chars,
            )
        )
        compact_payload.update({"frozen": True, "reranker_arm": arm})
        records.append(
            {
                "task_id": task.task_id,
                "arguments": {
                    "query": query or topic,
                    "subject": row.get("subject"),
                    "grade": row.get("grade"),
                    "mode": "or",
                },
                "image_evidence": {
                    "image_evidence": [],
                    "question": task.question,
                    "topic": topic,
                    "unknown_concepts": [],
                },
                "payload": compact_payload,
                "retrieval_conflict": None,
                "order_changes": False,
                "reranker_arm": arm,
                "rerankable": bool(source_rankings),
            }
        )
    return records


class FrozenRerankerSolver(FrozenContextOrderSolver):
    """Answer from one precomputed reranker arm without another retrieval call."""

    def __init__(
        self,
        settings: Settings,
        *,
        records: list[dict[str, Any]],
        arm: str,
        llm: Any | None = None,
    ) -> None:
        super().__init__(settings, records=records, order="score", llm=llm)
        self.arm = arm

    def solve(self, task: Task) -> SolveResult:
        result = super().solve(task)
        generation = dict(result.generation)
        generation.update(
            {
                "retrieval_strategy": f"{self.arm}_frozen_top_k",
                "reranker_arm": self.arm,
                "agent_strategy": "frozen_reranker_ablation_v1",
                "experiment_id": "reranker_accuracy_v1",
            }
        )
        return result.model_copy(
            update={
                "condition": f"agent_rag_frozen_{self.arm}",
                "generation": generation,
            }
        )


def run_frozen_reranker_generation(
    *,
    tasks: list[Task],
    records: list[dict[str, Any]],
    arm: str,
    out_path: Path,
    settings: Settings,
    retry_errors: bool,
) -> tuple[int, int]:
    solver = FrozenRerankerSolver(settings, records=records, arm=arm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path, retry_errors=retry_errors)
    todo = [task for task in tasks if task.task_id not in done]
    print(
        f"Задач: {len(tasks)}, уже готово: {len(tasks) - len(todo)}, "
        f"к прогону: {len(todo)}, arm={arm}"
    )

    errors = 0
    write_lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as destination:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(solver.solve, task): task for task in todo}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result.error:
                    errors += 1
                with write_lock:
                    destination.write(result.model_dump_json() + "\n")
                    destination.flush()
                print(
                    f"[{index}/{len(todo)}] {result.task_id}: "
                    f"{result.error or 'ok'} ({result.usage.latency_s}s)"
                )
    print(f"Готово: {len(todo)} прогнано, ошибок: {errors}, результат: {out_path}")
    return len(todo), errors


def _judge_verdicts(path: Path) -> dict[str, bool]:
    verdicts: dict[str, bool] = {}
    for row in _read_jsonl(path):
        task_id = str(row.get("task_id") or "")
        verdict = row.get("verdict")
        strict = verdict.get("strict_correct") if isinstance(verdict, dict) else None
        if not task_id or not isinstance(strict, bool):
            raise ValueError(f"{path}: invalid judge row for task_id={task_id!r}")
        if task_id in verdicts:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        verdicts[task_id] = strict
    return verdicts


def _accuracy(verdicts: dict[str, bool], task_ids: set[str]) -> dict[str, Any]:
    selected = [verdicts[task_id] for task_id in sorted(task_ids)]
    correct = sum(selected)
    total = len(selected)
    accuracy = correct / total if total else None
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_percent": round(100.0 * accuracy, 2) if accuracy is not None else None,
    }


def _mcnemar_exact_p(fixed: int, regressed: int) -> float:
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    smaller = min(fixed, regressed)
    one_tail = sum(math.comb(discordant, value) for value in range(smaller + 1))
    probability = one_tail / (2**discordant)
    return min(1.0, 2.0 * probability)


def _paired(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    task_ids: set[str],
) -> dict[str, Any]:
    fixed_ids: list[str] = []
    regressed_ids: list[str] = []
    unchanged_correct = 0
    unchanged_incorrect = 0
    for task_id in sorted(task_ids):
        before = baseline[task_id]
        after = candidate[task_id]
        if not before and after:
            fixed_ids.append(task_id)
        elif before and not after:
            regressed_ids.append(task_id)
        elif before:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1
    fixed = len(fixed_ids)
    regressed = len(regressed_ids)
    return {
        "paired": len(task_ids),
        "fixed": fixed,
        "regressed": regressed,
        "net_fixes": fixed - regressed,
        "unchanged_correct": unchanged_correct,
        "unchanged_incorrect": unchanged_incorrect,
        "mcnemar_exact_p": _mcnemar_exact_p(fixed, regressed),
        "fixed_task_ids": fixed_ids,
        "regressed_task_ids": regressed_ids,
    }


def summarize_judges(
    *,
    judge_paths: dict[str, Path],
    rerankable_task_ids: set[str],
) -> dict[str, Any]:
    if "dense" not in judge_paths:
        raise ValueError("dense judge is required as the paired baseline")
    verdicts = {arm: _judge_verdicts(path) for arm, path in judge_paths.items()}
    evaluated_ids = set(verdicts["dense"])
    for arm, arm_verdicts in verdicts.items():
        if set(arm_verdicts) != evaluated_ids:
            raise ValueError(f"{arm}: judge task IDs differ from dense")
    rerankable_evaluated = evaluated_ids & rerankable_task_ids
    unchanged_input_control = evaluated_ids - rerankable_task_ids

    accuracy = {
        arm: {
            "all_evaluated": _accuracy(arm_verdicts, evaluated_ids),
            "rerankable_only": _accuracy(arm_verdicts, rerankable_evaluated),
            "unchanged_input_control": _accuracy(
                arm_verdicts,
                unchanged_input_control,
            ),
        }
        for arm, arm_verdicts in verdicts.items()
    }
    comparisons: dict[str, Any] = {}
    for arm, arm_verdicts in verdicts.items():
        if arm == "dense":
            continue
        comparisons[arm] = {
            "all_evaluated": _paired(
                verdicts["dense"], arm_verdicts, evaluated_ids
            ),
            "rerankable_only": _paired(
                verdicts["dense"], arm_verdicts, rerankable_evaluated
            ),
            "unchanged_input_control": _paired(
                verdicts["dense"], arm_verdicts, unchanged_input_control
            ),
        }
    return {
        "schema_version": "reranker-accuracy-summary-v1",
        "baseline": "dense",
        "evaluated_tasks": len(evaluated_ids),
        "rerankable_evaluated_tasks": len(rerankable_evaluated),
        "unchanged_input_control_tasks": len(unchanged_input_control),
        "accuracy": accuracy,
        "comparisons_vs_dense": comparisons,
    }


def _prepare_command(args: argparse.Namespace) -> int:
    arms = (*DEFAULT_ARMS, "qwen3_reranker_06b") if args.include_qwen else DEFAULT_ARMS
    tasks = load_tasks(args.tasks)
    source_manifest, rows_by_arm = load_rankings_archive(
        args.rankings_zip,
        arms=arms,
    )
    indexed, rerankable = validate_ranking_arms(
        tasks=tasks,
        rows_by_arm=rows_by_arm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        records = build_context_records(
            tasks=tasks,
            rankings_by_task=indexed[arm],
            arm=arm,
            top_k=args.top_k,
            max_text_chars=args.max_text_chars,
        )
        _write_jsonl(args.output_dir / f"contexts_{arm}.jsonl", records)

    preparation = {
        "schema_version": "reranker-accuracy-preparation-v1",
        "rankings_zip": str(args.rankings_zip.resolve()),
        "rankings_zip_sha256": _sha256(args.rankings_zip),
        "source_manifest": source_manifest,
        "tasks": len(tasks),
        "rerankable_tasks": len(rerankable),
        "rerankable_task_ids": sorted(rerankable),
        "arms": list(arms),
        "top_k": args.top_k,
        "max_text_chars": args.max_text_chars,
    }
    manifest_path = args.output_dir / "preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(preparation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "rerankable_tasks": len(rerankable),
                "arms": list(arms),
                "top_k": args.top_k,
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    tasks = load_tasks(args.tasks)
    if args.limit is not None and args.limit > 0:
        tasks = tasks[: args.limit]
    records = _read_jsonl(args.contexts)
    _, errors = run_frozen_reranker_generation(
        tasks=tasks,
        records=records,
        arm=args.arm,
        out_path=args.output,
        settings=settings,
        retry_errors=args.retry_errors,
    )
    return 1 if errors else 0


def _parse_judge_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or not arm or not raw_path:
            raise ValueError("--judge must use ARM=PATH")
        if arm in paths:
            raise ValueError(f"duplicate judge arm: {arm}")
        paths[arm] = Path(raw_path)
    return paths


def _summarize_command(args: argparse.Namespace) -> int:
    preparation = json.loads(args.preparation_manifest.read_text(encoding="utf-8"))
    rerankable = set(preparation.get("rerankable_task_ids") or [])
    summary = summarize_judges(
        judge_paths=_parse_judge_paths(args.judge),
        rerankable_task_ids=rerankable,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="validate a reranker archive and materialize frozen top-k contexts",
    )
    prepare.add_argument("--tasks", type=Path, required=True)
    prepare.add_argument("--rankings-zip", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--top-k", type=int, default=5)
    prepare.add_argument("--max-text-chars", type=int, default=6_000)
    prepare.add_argument("--include-qwen", action="store_true")
    prepare.set_defaults(handler=_prepare_command)

    run = subparsers.add_parser(
        "run",
        help="generate answers from one frozen reranker context arm",
    )
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--contexts", type=Path, required=True)
    run.add_argument("--arm", choices=tuple(RANKING_FILES), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(handler=_run_command)

    summarize = subparsers.add_parser(
        "summarize",
        help="compare strict judge accuracy and paired changes against dense",
    )
    summarize.add_argument("--preparation-manifest", type=Path, required=True)
    summarize.add_argument(
        "--judge",
        action="append",
        required=True,
        help="ARM=PATH; repeat once per evaluated arm",
    )
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(handler=_summarize_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

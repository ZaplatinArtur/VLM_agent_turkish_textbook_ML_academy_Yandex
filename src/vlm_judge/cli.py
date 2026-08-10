from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .adjudication import build_adjudication_context
from .aggregation import aggregate_results
from .annotation_dataset import prepare_pointwise_annotation_dataset, sample_calibration_responses
from .arena import prepare_pairwise
from .assets import verify_benchmark_assets
from .backends import OpenAICompatibleBackend
from .calibration import prepare_calibration
from .calibration_analysis import analyze_arena_annotations, analyze_calibration
from .corpus import prepare_corpus
from .dryrun import run_synthetic_experiment
from .gold import apply_verified_gold
from .metrics import deterministic_match
from .ingest import import_candidates, read_records
from .judge_audit import audit_judge_run, validate_judge_completion
from .mla_adapter import (
    build_seed_text_tasks,
    prepare_image_judge_input,
    prepare_text_judge_input,
)
from .pipeline import prepare_request_records
from .retrieval import build_bm25_index, index_info, search_bm25
from .retrieval_eval import evaluate_retrieval, prepare_qrels_template
from .retrieval_server import serve_retrieval
from .runner import evaluate_items
from .schema import EvaluationItem
from .sources import prepare_sources
from .text_judge import evaluate_text_records
from .validation import parse_run_specs, validate_experiment_runs
from .validation_archive import (
    build_image_only_validation_tasks,
    build_validation_manifest,
    build_validation_seed_manifest,
    build_validation_tasks,
    extract_validation_archive,
    extract_validation_text,
)


def _prepare_sources(args: argparse.Namespace) -> int:
    inventory = prepare_sources(
        Path(args.main_workbook),
        Path(args.math_workbook),
        Path(args.corpus),
        Path(args.output_dir),
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


def _score_deterministic(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = EvaluationItem.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid item on line {line_number}: {exc}") from exc
            result = deterministic_match(
                item.reference_answer,
                item.candidate_answer,
                item.answer_type,
                acceptable_answers=item.acceptable_answers,
                numeric_tolerance=args.numeric_tolerance,
            )
            destination.write(
                json.dumps(
                    {
                        "task_id": item.task_id,
                        "setup": item.setup,
                        "deterministic": {
                            "applicable": result.applicable,
                            "matched": result.matched,
                            "method": result.method,
                            "normalized_reference": result.normalized_reference,
                            "normalized_candidate": result.normalized_candidate,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0


def _prepare_requests(args: argparse.Namespace) -> int:
    count = prepare_request_records(
        Path(args.input),
        Path(args.output),
        prompt_version=args.prompt_version,
    )
    print(json.dumps({"prepared_requests": count, "output": args.output}, ensure_ascii=False))
    return 0


def _aggregate_record_key(record: dict[str, Any]) -> tuple[str, str]:
    task_id = str(record.get("task_id") or "").strip()
    setup = str(record.get("setup") or "").strip()
    if not task_id or not setup:
        raise ValueError("aggregate records require non-empty task_id and setup")
    return task_id, setup


def _load_aggregate_records(
    input_paths: list[Path],
    overlay_paths: list[Path],
) -> tuple[list[dict[str, Any]], int]:
    records = [
        record
        for path in input_paths
        for record in read_records(path)
    ]
    if not overlay_paths:
        return records, 0

    positions: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        key = _aggregate_record_key(record)
        if key in positions:
            raise ValueError(
                f"cannot apply overlays to duplicate base aggregate unit: {key[0]}::{key[1]}"
            )
        positions[key] = index

    replacements = 0
    seen_overlay_keys: set[tuple[str, str]] = set()
    for path in overlay_paths:
        for record in read_records(path):
            key = _aggregate_record_key(record)
            if key in seen_overlay_keys:
                raise ValueError(f"duplicate aggregate overlay unit: {key[0]}::{key[1]}")
            seen_overlay_keys.add(key)
            if key not in positions:
                raise ValueError(f"aggregate overlay has no base unit: {key[0]}::{key[1]}")
            records[positions[key]] = record
            replacements += 1
    return records, replacements


def _aggregate(args: argparse.Namespace) -> int:
    input_paths = [Path(value) for value in args.input]
    overlay_paths = [Path(value) for value in args.overlay]
    records, replacements = _load_aggregate_records(input_paths, overlay_paths)
    result = aggregate_results(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "overlay_replacements": replacements,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _prepare_calibration(args: argparse.Namespace) -> int:
    result = prepare_calibration(
        Path(args.benchmark),
        Path(args.output_dir),
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _verify_assets(args: argparse.Namespace) -> int:
    result = verify_benchmark_assets(
        Path(args.benchmark),
        Path(args.manifest),
        Path(args.summary),
        workers=args.workers,
        retries=args.retries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _import_candidates(args: argparse.Namespace) -> int:
    result = import_candidates(
        Path(args.benchmark),
        Path(args.responses),
        Path(args.output),
        setup=args.setup,
        id_field=args.id_field,
        answer_field=args.answer_field,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _prepare_mla_judge_input(args: argparse.Namespace) -> int:
    report = prepare_text_judge_input(
        Path(args.tasks),
        Path(args.results),
        Path(args.output),
        require_all=args.require_all,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _prepare_image_judge_input(args: argparse.Namespace) -> int:
    report = prepare_image_judge_input(
        Path(args.manifest),
        Path(args.results),
        Path(args.data_root),
        Path(args.output),
        require_all=args.require_all,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_seed_text_tasks(args: argparse.Namespace) -> int:
    report = build_seed_text_tasks(Path(args.input), Path(args.output))
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _prepare_validation_archive(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    extraction = extract_validation_archive(
        Path(args.archive), output_dir, reuse_existing=args.reuse_extracted
    )
    manifest = build_validation_manifest(
        Path(extraction["workbook"]),
        output_dir,
        output_dir / "validation_manifest.jsonl",
    )
    seed_manifest = build_validation_seed_manifest(
        Path(extraction["workbook"]),
        output_dir,
        output_dir / "seed_manifest_8.jsonl",
    )
    print(
        json.dumps(
            {"extraction": extraction, "manifest": manifest, "seed_manifest": seed_manifest},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _extract_validation_text(args: argparse.Namespace) -> int:
    backend = OpenAICompatibleBackend(
        args.base_url,
        args.model,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        timeout=args.timeout,
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        use_response_format=not args.no_response_format,
        enable_thinking=False,
    )
    report = extract_validation_text(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.output),
        backend,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_validation_tasks(args: argparse.Namespace) -> int:
    report = build_validation_tasks(
        Path(args.manifest),
        Path(args.extractions),
        Path(args.output),
        require_all=args.require_all,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_image_validation_tasks(args: argparse.Namespace) -> int:
    report = build_image_only_validation_tasks(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _prepare_pairs(args: argparse.Namespace) -> int:
    result = prepare_pairwise(
        [Path(value) for value in args.input],
        Path(args.output),
        setup_a=args.setup_a,
        setup_b=args.setup_b,
        seed=args.seed,
        mirrored=args.mirrored,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _write_report(value: dict, output_value: str) -> None:
    output = Path(output_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _analyze_calibration(args: argparse.Namespace) -> int:
    human_records = [record for value in args.human for record in read_records(Path(value))]
    judge_records = read_records(Path(args.judge))
    report = analyze_calibration(human_records, judge_records)
    _write_report(report, args.output)
    print(json.dumps({"matched": report["matched_comparisons"], "output": args.output}, ensure_ascii=False))
    return 0


def _analyze_arena(args: argparse.Namespace) -> int:
    records = [record for value in args.annotations for record in read_records(Path(value))]
    report = analyze_arena_annotations(records)
    _write_report(report, args.output)
    print(json.dumps({"votes": report["complete_votes"], "output": args.output}, ensure_ascii=False))
    return 0


def _prepare_corpus(args: argparse.Namespace) -> int:
    report = prepare_corpus(
        Path(args.input),
        Path(args.output_dir),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_bm25(args: argparse.Namespace) -> int:
    report = build_bm25_index(
        Path(args.chunks),
        Path(args.index),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _search_bm25(args: argparse.Namespace) -> int:
    result = search_bm25(
        Path(args.index),
        args.query,
        top_k=args.top_k,
        subject=args.subject,
        grade=args.grade,
        mode=args.mode,
        low_information_weight=args.low_information_weight,
    )
    if args.output:
        _write_report(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _inspect_bm25(args: argparse.Namespace) -> int:
    print(json.dumps(index_info(Path(args.index)), ensure_ascii=False, indent=2))
    return 0


def _serve_retrieval(args: argparse.Namespace) -> int:
    serve_retrieval(Path(args.index), host=args.host, port=args.port)
    return 0


def _evaluate_retrieval(args: argparse.Namespace) -> int:
    report = evaluate_retrieval(
        Path(args.index),
        Path(args.qrels),
        ks=args.k or (1, 5, 10),
        mode=args.mode,
        low_information_weight=args.low_information_weight,
    )
    _write_report(report, args.output)
    print(json.dumps({"queries": report["queries"], "metrics": report["metrics"], "output": args.output}, ensure_ascii=False))
    return 0


def _prepare_retrieval_qrels(args: argparse.Namespace) -> int:
    report = prepare_qrels_template(Path(args.benchmark), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _validate_runs(args: argparse.Namespace) -> int:
    runs = parse_run_specs(args.run)
    report = validate_experiment_runs(
        Path(args.benchmark),
        runs,
        strict_metadata=args.strict_metadata,
    )
    _write_report(report, args.output)
    print(
        json.dumps(
            {
                "ready_for_experiment": report["ready_for_experiment"],
                "errors": report["error_count"],
                "warnings": report["warning_count"],
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ready_for_experiment"] else 2


def _synthetic_dry_run(args: argparse.Namespace) -> int:
    report = run_synthetic_experiment(
        Path(args.benchmark),
        Path(args.output_dir),
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run_judge(args: argparse.Namespace) -> int:
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {args.api_key_env!r} is not set")
    backend = OpenAICompatibleBackend(
        args.base_url,
        args.model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        use_response_format=not args.no_response_format,
        image_mode=args.image_mode,
        image_cache_dir=Path(args.image_cache_dir),
        enable_thinking=False if args.disable_thinking else None,
        provider=args.provider,
    )
    items = [EvaluationItem.from_dict(record) for record in read_records(Path(args.input))]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        items = items[: args.limit]
    count = evaluate_items(
        items,
        backend,
        Path(args.output),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        prompt_version=args.prompt_version,
        max_attempts=args.max_attempts,
        workers=args.workers,
        retry_delay_seconds=args.retry_delay,
    )
    print(json.dumps({"evaluated": count, "output": args.output}, ensure_ascii=False))
    return 0


def _run_text_judge(args: argparse.Namespace) -> int:
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {args.api_key_env!r} is not set")
    backend = OpenAICompatibleBackend(
        args.base_url,
        args.model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        use_response_format=not args.no_response_format,
        enable_thinking=False,
        provider=args.provider,
        image_mode="url",
    )
    records = read_records(Path(args.input))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        records = records[: args.limit]
    report = evaluate_text_records(
        records,
        backend,
        Path(args.output),
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay,
        retry_failures=args.retry_failures,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 2


def _prepare_adjudication(args: argparse.Namespace) -> int:
    tasks = read_records(Path(args.dataset))
    judge_results = read_records(Path(args.judge))
    human_annotations = [
        record for value in args.human for record in read_records(Path(value))
    ]
    existing = read_records(Path(args.adjudications)) if args.adjudications else []
    context = build_adjudication_context(
        tasks,
        judge_results,
        human_annotations,
        existing,
        low_confidence_threshold=args.low_confidence_threshold,
        agreement_sample_rate=args.agreement_sample_rate,
        agreement_sample_seed=args.agreement_sample_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in context["items"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    _write_report(
        {"enabled": context["enabled"], "stats": context["stats"], "config": context["config"]},
        args.summary,
    )
    print(json.dumps({"queue_items": len(context["items"]), "output": args.output, "summary": args.summary}, ensure_ascii=False))
    return 0


def _apply_gold(args: argparse.Namespace) -> int:
    report = apply_verified_gold(
        Path(args.dataset),
        Path(args.gold),
        Path(args.output),
        require_all=args.require_all,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _prepare_pointwise_ui(args: argparse.Namespace) -> int:
    report = prepare_pointwise_annotation_dataset(
        [Path(value) for value in args.input],
        Path(args.output),
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _audit_judge_run(args: argparse.Namespace) -> int:
    report = audit_judge_run(read_records(Path(args.input)))
    _write_report(report, args.output)
    print(
        json.dumps(
            {
                "records": report["records"],
                "schema_valid_rate": report["schema_valid_rate_after_retries"],
                "failed": report["failed_records"],
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _verify_judge_output(args: argparse.Namespace) -> int:
    report = validate_judge_completion(
        read_records(Path(args.expected)),
        read_records(Path(args.judge)),
    )
    _write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 2


def _sample_calibration_responses(args: argparse.Namespace) -> int:
    report = sample_calibration_responses(
        [Path(value) for value in args.input],
        Path(args.output),
        size=args.size,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-judge")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-sources", help="normalize workbooks and inventory corpus")
    prepare.add_argument("--main-workbook", required=True)
    prepare.add_argument("--math-workbook", required=True)
    prepare.add_argument("--corpus", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(handler=_prepare_sources)

    score = commands.add_parser("score-deterministic", help="score JSONL candidates without an LLM")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--numeric-tolerance", type=float, default=1e-9)
    score.set_defaults(handler=_score_deterministic)

    requests = commands.add_parser("prepare-requests", help="build blinded multimodal judge requests")
    requests.add_argument("--input", required=True)
    requests.add_argument("--output", required=True)
    requests.add_argument("--prompt-version", default="judge-v2")
    requests.set_defaults(handler=_prepare_requests)

    aggregate = commands.add_parser("aggregate", help="aggregate scored records by setup and subject")
    aggregate.add_argument(
        "--input",
        action="append",
        required=True,
        help="scored JSONL input; repeat to aggregate multiple setups",
    )
    aggregate.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="corrected JSONL rows that replace matching task_id/setup units",
    )
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=_aggregate)

    calibration = commands.add_parser("prepare-calibration", help="select calibration tasks")
    calibration.add_argument("--benchmark", required=True)
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--seed", default="calibration-v1")
    calibration.set_defaults(handler=_prepare_calibration)

    assets = commands.add_parser("verify-assets", help="verify Yandex public image links")
    assets.add_argument("--benchmark", required=True)
    assets.add_argument("--manifest", required=True)
    assets.add_argument("--summary", required=True)
    assets.add_argument("--workers", type=int, default=12)
    assets.add_argument("--retries", type=int, default=1)
    assets.set_defaults(handler=_verify_assets)

    candidate_import = commands.add_parser("import-candidates", help="attach agent outputs to benchmark tasks")
    candidate_import.add_argument("--benchmark", required=True)
    candidate_import.add_argument("--responses", required=True)
    candidate_import.add_argument("--output", required=True)
    candidate_import.add_argument("--setup", required=True, choices=["no_tools", "web_search", "textbook_retrieval", "unknown"])
    candidate_import.add_argument("--id-field", default="task_id")
    candidate_import.add_argument("--answer-field")
    candidate_import.set_defaults(handler=_import_candidates)

    mla_judge_input = commands.add_parser(
        "prepare-mla-judge-input",
        help="join mla_baseline tasks and solver results for the text binary judge",
    )
    mla_judge_input.add_argument("--tasks", required=True)
    mla_judge_input.add_argument("--results", required=True)
    mla_judge_input.add_argument("--output", required=True)
    mla_judge_input.add_argument("--require-all", action="store_true")
    mla_judge_input.set_defaults(handler=_prepare_mla_judge_input)

    image_judge_input = commands.add_parser(
        "prepare-image-judge-input",
        help="join solver results to original validation question/reference images",
    )
    image_judge_input.add_argument("--manifest", required=True)
    image_judge_input.add_argument("--results", required=True)
    image_judge_input.add_argument("--data-root", required=True)
    image_judge_input.add_argument("--output", required=True)
    image_judge_input.add_argument("--require-all", action="store_true")
    image_judge_input.set_defaults(handler=_prepare_image_judge_input)

    seed_tasks = commands.add_parser(
        "build-seed-text-tasks",
        help="extract text-only MLA tasks from the real legacy failure sample",
    )
    seed_tasks.add_argument("--input", required=True)
    seed_tasks.add_argument("--output", required=True)
    seed_tasks.set_defaults(handler=_build_seed_text_tasks)

    validation_archive = commands.add_parser(
        "prepare-validation-archive",
        help="extract the updated mentor ZIP and resolve local Sheet1 question/reference images",
    )
    validation_archive.add_argument("--archive", required=True)
    validation_archive.add_argument("--output-dir", required=True)
    validation_archive.add_argument("--reuse-extracted", action="store_true")
    validation_archive.set_defaults(handler=_prepare_validation_archive)

    validation_extract = commands.add_parser(
        "extract-validation-text",
        help="transcribe local validation questions and reference images with multimodal Qwen",
    )
    validation_extract.add_argument("--manifest", required=True)
    validation_extract.add_argument("--data-root", required=True)
    validation_extract.add_argument("--output", required=True)
    validation_extract.add_argument("--base-url", required=True)
    validation_extract.add_argument("--model", required=True)
    validation_extract.add_argument("--api-key-env")
    validation_extract.add_argument("--timeout", type=float, default=300.0)
    validation_extract.add_argument("--max-tokens", type=int, default=3072)
    validation_extract.add_argument("--seed", type=int, default=20260721)
    validation_extract.add_argument("--workers", type=int, default=4)
    validation_extract.add_argument("--limit", type=int)
    validation_extract.add_argument("--no-response-format", action="store_true")
    validation_extract.set_defaults(handler=_extract_validation_text)

    validation_tasks = commands.add_parser(
        "build-validation-tasks",
        help="join the validation manifest and Qwen transcriptions into MLA tasks",
    )
    validation_tasks.add_argument("--manifest", required=True)
    validation_tasks.add_argument("--extractions", required=True)
    validation_tasks.add_argument("--output", required=True)
    validation_tasks.add_argument("--require-all", action="store_true")
    validation_tasks.set_defaults(handler=_build_validation_tasks)

    image_validation_tasks = commands.add_parser(
        "build-image-validation-tasks",
        help="build MLA tasks from every original question image without OCR",
    )
    image_validation_tasks.add_argument("--manifest", required=True)
    image_validation_tasks.add_argument("--data-root", required=True)
    image_validation_tasks.add_argument("--output", required=True)
    image_validation_tasks.set_defaults(handler=_build_image_validation_tasks)

    pairs = commands.add_parser("prepare-pairs", help="build blinded LMArena-style A/B records")
    pairs.add_argument("--input", action="append", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--setup-a", required=True)
    pairs.add_argument("--setup-b", required=True)
    pairs.add_argument("--seed", default="arena-v1")
    pairs.add_argument("--mirrored", action="store_true")
    pairs.set_defaults(handler=_prepare_pairs)

    calibration_report = commands.add_parser(
        "analyze-calibration",
        help="compare human labels with LLM-judge verdicts",
    )
    calibration_report.add_argument("--human", action="append", required=True)
    calibration_report.add_argument("--judge", required=True)
    calibration_report.add_argument("--output", required=True)
    calibration_report.set_defaults(handler=_analyze_calibration)

    arena_report = commands.add_parser(
        "analyze-arena",
        help="audit mirrored A/B votes for position bias",
    )
    arena_report.add_argument("--annotations", action="append", required=True)
    arena_report.add_argument("--output", required=True)
    arena_report.set_defaults(handler=_analyze_arena)

    corpus = commands.add_parser(
        "prepare-corpus",
        help="deduplicate pages and build provenance-preserving text/image chunks",
    )
    corpus.add_argument("--input", required=True)
    corpus.add_argument("--output-dir", required=True)
    corpus.add_argument("--max-chars", type=int, default=1600)
    corpus.add_argument("--overlap-chars", type=int, default=200)
    corpus.set_defaults(handler=_prepare_corpus)

    bm25 = commands.add_parser("build-bm25", help="build a SQLite FTS5/BM25 index over text chunks")
    bm25.add_argument("--chunks", required=True)
    bm25.add_argument("--index", required=True)
    bm25.add_argument("--batch-size", type=int, default=1000)
    bm25.set_defaults(handler=_build_bm25)

    search = commands.add_parser("search-bm25", help="query the lexical retrieval baseline")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--subject")
    search.add_argument("--grade")
    search.add_argument("--mode", choices=["or", "and"], default="or")
    search.add_argument("--low-information-weight", type=float, default=0.25)
    search.add_argument("--output")
    search.set_defaults(handler=_search_bm25)

    inspect_index = commands.add_parser("inspect-bm25", help="show lexical index metadata")
    inspect_index.add_argument("--index", required=True)
    inspect_index.set_defaults(handler=_inspect_bm25)

    retrieval_api = commands.add_parser("serve-retrieval", help="serve BM25 as a local agent tool")
    retrieval_api.add_argument("--index", required=True)
    retrieval_api.add_argument("--host", default="127.0.0.1")
    retrieval_api.add_argument("--port", type=int, default=8770)
    retrieval_api.set_defaults(handler=_serve_retrieval)

    qrels_template = commands.add_parser(
        "prepare-retrieval-qrels",
        help="create a task-aligned relevance-annotation template for the retrieval team",
    )
    qrels_template.add_argument("--benchmark", required=True)
    qrels_template.add_argument("--output", required=True)
    qrels_template.set_defaults(handler=_prepare_retrieval_qrels)

    retrieval_eval = commands.add_parser(
        "evaluate-retrieval",
        help="measure hit rate, recall, MRR, and latency against retrieval qrels",
    )
    retrieval_eval.add_argument("--index", required=True)
    retrieval_eval.add_argument("--qrels", required=True)
    retrieval_eval.add_argument("--output", required=True)
    retrieval_eval.add_argument("--k", action="append", type=int)
    retrieval_eval.add_argument("--mode", choices=["or", "and"], default="or")
    retrieval_eval.add_argument("--low-information-weight", type=float, default=0.25)
    retrieval_eval.set_defaults(handler=_evaluate_retrieval)

    validation = commands.add_parser("validate-runs", help="audit the three setup files before evaluation")
    validation.add_argument("--benchmark", required=True)
    validation.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SETUP=PATH",
        help="repeat for no_tools, web_search, and textbook_retrieval",
    )
    validation.add_argument("--output", required=True)
    validation.add_argument("--strict-metadata", action="store_true", help="treat missing run/retrieval provenance as an error")
    validation.set_defaults(handler=_validate_runs)

    dry_run = commands.add_parser(
        "synthetic-dry-run",
        help="exercise the complete three-setup pipeline without a model",
    )
    dry_run.add_argument("--benchmark", required=True)
    dry_run.add_argument("--output-dir", required=True)
    dry_run.add_argument("--seed", default="synthetic-experiment-v1")
    dry_run.add_argument("--limit", type=int)
    dry_run.set_defaults(handler=_synthetic_dry_run)

    judge = commands.add_parser(
        "run-judge",
        help="run the blinded judge through OpenRouter or another OpenAI-compatible endpoint",
    )
    judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--api-key-env")
    judge.add_argument("--provider", choices=["vllm", "openrouter"], default="vllm")
    judge.add_argument("--timeout", type=float, default=120.0)
    judge.add_argument("--temperature", type=float, default=0.0)
    judge.add_argument("--max-tokens", type=int, default=900)
    judge.add_argument("--seed", type=int, default=20260714)
    judge.add_argument("--image-mode", choices=["url", "data_url"], default="url")
    judge.add_argument("--image-cache-dir", default="artifacts/cache/judge_images")
    judge.add_argument("--no-response-format", action="store_true")
    judge.add_argument(
        "--disable-thinking",
        action="store_true",
        help="ask compatible Qwen chat templates to skip hidden reasoning",
    )
    judge.add_argument("--cache-dir", default="artifacts/cache/judge")
    judge.add_argument("--prompt-version", default="judge-v2")
    judge.add_argument("--max-attempts", type=int, default=2)
    judge.add_argument("--workers", type=int, default=1, help="parallel endpoint requests; increase only after the 10-case smoke test")
    judge.add_argument("--retry-delay", type=float, default=1.0, help="linear backoff base in seconds between attempts")
    judge.add_argument("--limit", type=int, help="evaluate only the first N records for a transport smoke test")
    judge.set_defaults(handler=_run_judge)

    text_judge = commands.add_parser(
        "run-text-judge",
        help="run the strict text-only binary judge through an OpenAI-compatible endpoint",
    )
    text_judge.add_argument("--input", required=True)
    text_judge.add_argument("--output", required=True)
    text_judge.add_argument("--base-url", required=True)
    text_judge.add_argument("--model", required=True)
    text_judge.add_argument("--api-key-env")
    text_judge.add_argument("--provider", choices=["vllm", "openrouter"], default="vllm")
    text_judge.add_argument("--timeout", type=float, default=120.0)
    text_judge.add_argument("--temperature", type=float, default=0.0)
    text_judge.add_argument("--max-tokens", type=int, default=256)
    text_judge.add_argument("--seed", type=int, default=20260714)
    text_judge.add_argument("--no-response-format", action="store_true")
    text_judge.add_argument("--max-attempts", type=int, default=2)
    text_judge.add_argument("--retry-delay", type=float, default=1.0)
    text_judge.add_argument("--limit", type=int)
    text_judge.add_argument(
        "--retry-failures",
        action="store_true",
        help="preserve successful output rows and retry only failed or missing task IDs",
    )
    text_judge.set_defaults(handler=_run_text_judge)

    adjudication = commands.add_parser(
        "prepare-adjudication",
        help="build a prioritized human-vs-LLM disagreement queue",
    )
    adjudication.add_argument("--dataset", required=True)
    adjudication.add_argument("--judge", required=True)
    adjudication.add_argument("--human", action="append", required=True)
    adjudication.add_argument("--adjudications")
    adjudication.add_argument("--output", required=True)
    adjudication.add_argument("--summary", required=True)
    adjudication.add_argument("--low-confidence-threshold", type=float, default=0.75)
    adjudication.add_argument("--agreement-sample-rate", type=float, default=0.10)
    adjudication.add_argument("--agreement-sample-seed", default="adjudication-control-v1")
    adjudication.set_defaults(handler=_prepare_adjudication)

    gold = commands.add_parser(
        "apply-gold",
        help="attach verified task-scoped gold transcriptions to judge input records",
    )
    gold.add_argument("--dataset", required=True)
    gold.add_argument("--gold", required=True)
    gold.add_argument("--output", required=True)
    gold.add_argument("--require-all", action="store_true")
    gold.set_defaults(handler=_apply_gold)

    pointwise_ui = commands.add_parser(
        "prepare-pointwise-ui",
        help="combine and shuffle setup runs for blinded human annotation",
    )
    pointwise_ui.add_argument("--input", action="append", required=True)
    pointwise_ui.add_argument("--output", required=True)
    pointwise_ui.add_argument("--seed", default="pointwise-ui-v1")
    pointwise_ui.set_defaults(handler=_prepare_pointwise_ui)

    judge_audit = commands.add_parser(
        "audit-judge-run",
        help="summarize judge validity, retries, cache, model identity, and token usage",
    )
    judge_audit.add_argument("--input", required=True)
    judge_audit.add_argument("--output", required=True)
    judge_audit.set_defaults(handler=_audit_judge_run)

    judge_verify = commands.add_parser(
        "verify-judge-output",
        help="require exact task coverage and valid error-free verdicts before analytics import",
    )
    judge_verify.add_argument("--expected", required=True)
    judge_verify.add_argument("--judge", required=True)
    judge_verify.add_argument("--output", required=True)
    judge_verify.set_defaults(handler=_verify_judge_output)

    response_sample = commands.add_parser(
        "sample-calibration-responses",
        help="select a setup-balanced stratified human calibration set after agent runs exist",
    )
    response_sample.add_argument("--input", action="append", required=True)
    response_sample.add_argument("--output", required=True)
    response_sample.add_argument("--size", type=int, default=120)
    response_sample.add_argument("--seed", default="response-calibration-v1")
    response_sample.set_defaults(handler=_sample_calibration_responses)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

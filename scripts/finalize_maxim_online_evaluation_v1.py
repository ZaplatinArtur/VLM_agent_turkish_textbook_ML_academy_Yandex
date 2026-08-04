#!/usr/bin/env python3
"""Fail-closed two-stage wrapper for one Maksim online evaluation mode.

``prepare`` delegates solver-to-queue work to
``prepare_maxim_online_evaluation_v1``.  ``finalize`` revalidates that prepare
manifest and every pinned input/artifact, checks the fresh judge-v2 result in
exact queue order, then delegates combination and scoring to the existing
merge/scorer modules.

Final artifacts are never overwritten by default.  ``--overwrite-final`` is
an explicit opt-in that moves old final artifacts into a recoverable archive
before producing a complete new result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Import the exact in-repository judge adapter used by the pinned remote
# judge.  Its source files are byte-pinned below before any request ID is
# trusted.
REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_JUDGE_SRC = REPO_ROOT / "src"
if str(FROZEN_JUDGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_JUDGE_SRC))

import merge_maxim_judge_v2_results as merger
import prepare_maxim_online_evaluation_v1 as preparation
import score_maxim_full274 as scorer
import vlm_judge.pipeline as frozen_pipeline_module
import vlm_judge.prompts as frozen_prompts_module
import vlm_judge.schema as frozen_schema_module
from vlm_judge.pipeline import request_id as frozen_judge_request_id
from vlm_judge.schema import EvaluationItem


SCHEMA_VERSION = "maxim-online-finalization-v1"
FROZEN_JUDGE_PROMPT_VERSION = "judge-v2"
FROZEN_JUDGE_MODEL = "Qwen/Qwen3.5-9B"
FROZEN_JUDGE_ADAPTER_SHA256 = {
    "prompts.py": "84a1342612ca1fce9db9b7d4adfc4215b3d24e77f60d49a47635a47cc190b8d5",
    "pipeline.py": "d61a248dd5d078924475c819ef824c5e2dc1e587683b2cd84844bb01f72f2f73",
    "schema.py": "d86fed78c297d0479d00ceb2d54fad502dfbfe8b49cd7be08fed2d25fc7a4e7c",
}
FROZEN_JUDGE_BACKEND_CONFIG = {
    "backend": "openai-compatible",
    "model": FROZEN_JUDGE_MODEL,
    "endpoint": "http://127.0.0.1:18005/v1/chat/completions",
    "temperature": 0.0,
    "max_tokens": 900,
    "seed": 20260714,
    "use_response_format": True,
    "enable_thinking": False,
    "image_mode": "data_url",
}
FROZEN_JUDGE_BACKEND_CONFIG_SHA256 = (
    "e3f71b4af7fa8ad8a6db755d43bdf4a895d087b701436c105da8c5416804fbd9"
)
FINALIZE_LOCK_NAME = ".finalize_maxim_online_evaluation_v1.lock"
FINAL_OUTPUT_NAMES = {
    "matched_judge": "matched_image97_judge.jsonl",
    "matched_manifest": "matched_image97_judge_manifest.json",
    "matched_checksums": "matched_image97_judge.sha256",
    "score_json": "score.json",
    "score_markdown": "score.md",
    "score_checksums": "score.sha256",
    "finalization_manifest": "finalization_manifest.json",
    "finalization_checksums": "finalization.sha256",
}


class FinalizationError(ValueError):
    """Raised when a staged evaluation cannot be finalized safely."""


class ExclusiveOutputDirLock:
    """Atomic, fail-closed interprocess lock for one output directory."""

    def __init__(self, output_dir: Path, *, mode: str, label: str) -> None:
        self.output_dir = output_dir.resolve()
        self.path = self.output_dir / FINALIZE_LOCK_NAME
        self.mode = mode
        self.label = label
        self.token = uuid.uuid4().hex
        self.file_descriptor: int | None = None

    def __enter__(self) -> "ExclusiveOutputDirLock":
        if not self.output_dir.is_dir():
            raise FinalizationError(
                f"prepared output directory does not exist: {self.output_dir}"
            )
        payload = (
            json.dumps(
                {
                    "schema_version": "maxim-online-finalize-lock-v1",
                    "token": self.token,
                    "pid": os.getpid(),
                    "mode": self.mode,
                    "label": self.label,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        try:
            self.file_descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                owner = "<unreadable>"
            raise FinalizationError(
                f"output directory is locked by another finalize process: "
                f"{self.path}; owner={owner[:500]}"
            ) from exc
        try:
            os.write(self.file_descriptor, payload)
            os.fsync(self.file_descriptor)
        except Exception:
            os.close(self.file_descriptor)
            self.file_descriptor = None
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
            self.file_descriptor = None
        owns_lock = False
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            owns_lock = record.get("token") == self.token
        except (OSError, json.JSONDecodeError, AttributeError):
            owns_lock = False
        if owns_lock:
            self.path.unlink()
        elif exc_type is None:
            raise FinalizationError(
                f"finalize lock ownership changed unexpectedly; refusing cleanup: {self.path}"
            )
        return False


def _canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_frozen_judge_adapter() -> dict[str, str]:
    """Pin both request construction code and the imported module locations."""

    modules = {
        "prompts.py": frozen_prompts_module,
        "pipeline.py": frozen_pipeline_module,
        "schema.py": frozen_schema_module,
    }
    verified: dict[str, str] = {}
    for name, module in modules.items():
        expected_path = (FROZEN_JUDGE_SRC / "vlm_judge" / name).resolve()
        module_path = Path(str(module.__file__)).resolve()
        if module_path != expected_path:
            raise FinalizationError(
                f"frozen judge adapter import mismatch for {name}: "
                f"expected={expected_path}, imported={module_path}"
            )
        actual = preparation.sha256_file(expected_path)
        expected = FROZEN_JUDGE_ADAPTER_SHA256[name]
        if actual != expected:
            raise FinalizationError(
                f"frozen judge adapter SHA256 mismatch for {name}: "
                f"expected={expected}, actual={actual}"
            )
        verified[name] = actual
    config_hash = _canonical_sha256(FROZEN_JUDGE_BACKEND_CONFIG)
    if config_hash != FROZEN_JUDGE_BACKEND_CONFIG_SHA256:
        raise FinalizationError(
            "internal frozen judge backend config hash mismatch; "
            f"expected={FROZEN_JUDGE_BACKEND_CONFIG_SHA256}, actual={config_hash}"
        )
    return verified


def expected_fresh_request_id(queue_row: dict[str, Any]) -> str:
    """Reproduce the exact frozen judge-v2 semantic request fingerprint."""

    try:
        item = EvaluationItem.from_dict(queue_row)
        return frozen_judge_request_id(item, FROZEN_JUDGE_PROMPT_VERSION)
    except (KeyError, TypeError, ValueError) as exc:
        task_id = str(queue_row.get("task_id") or "<missing>")
        raise FinalizationError(
            f"fresh queue {task_id}: frozen judge adapter rejected row: {exc}"
        ) from exc


def validate_fresh_judge_request_linkage(
    *, result: dict[str, Any], queue_row: dict[str, Any]
) -> dict[str, str]:
    """Bind one verdict to the exact queue request and frozen backend config."""

    task_id = str(queue_row.get("task_id") or "<missing>")
    expected_request = expected_fresh_request_id(queue_row)
    actual_request = str(result.get("request_id") or "").strip()
    if actual_request != expected_request:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: request_id does not match exact "
            f"prepared candidate request; expected={expected_request}, "
            f"actual={actual_request or '<missing>'}"
        )

    for field in ("setup", "subject", "grade", "answer_type", "metadata"):
        if result.get(field) != queue_row.get(field):
            raise FinalizationError(
                f"fresh judge-v2 result {task_id}: output field {field!r} "
                "differs from prepared queue"
            )

    judge = result.get("judge")
    if not isinstance(judge, dict):
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: missing judge provenance"
        )
    if judge.get("backend") != FROZEN_JUDGE_BACKEND_CONFIG["backend"]:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: wrong judge backend"
        )
    if judge.get("model") != FROZEN_JUDGE_MODEL:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: wrong configured judge model"
        )
    backend_config = judge.get("backend_config")
    if backend_config != FROZEN_JUDGE_BACKEND_CONFIG:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: backend_config differs from "
            "frozen judge settings"
        )
    config_hash = _canonical_sha256(backend_config)
    if config_hash != FROZEN_JUDGE_BACKEND_CONFIG_SHA256:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: frozen backend config hash mismatch"
        )
    if judge.get("backend_config_hash") != config_hash:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: recorded backend_config_hash mismatch"
        )
    expected_cache_key = hashlib.sha256(
        f"{expected_request}:{config_hash}".encode("ascii")
    ).hexdigest()
    if judge.get("cache_key") != expected_cache_key:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: cache_key does not bind the exact "
            "request_id and backend config"
        )
    response_metadata = judge.get("response_metadata")
    served_model = (
        response_metadata.get("served_model")
        if isinstance(response_metadata, dict)
        else None
    )
    if served_model != FROZEN_JUDGE_MODEL:
        raise FinalizationError(
            f"fresh judge-v2 result {task_id}: served_model is not "
            f"{FROZEN_JUDGE_MODEL!r}"
        )
    return {
        "request_id": expected_request,
        "backend_config_hash": config_hash,
        "cache_key": expected_cache_key,
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"{label}: file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise FinalizationError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label}: expected a JSON object")
    return value


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise FinalizationError(f"{label}: missing non-empty value")
    return text


def _same_path(actual: Any, expected: Path, label: str) -> None:
    actual_path = Path(_require_text(actual, f"{label}.path")).resolve()
    if actual_path != expected.resolve():
        raise FinalizationError(
            f"{label}: path mismatch; expected={expected.resolve()}, actual={actual_path}"
        )


def _verify_recorded_file(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise FinalizationError(f"{label}: missing manifest record")
    path = Path(_require_text(record.get("path"), f"{label}.path"))
    if expected_path is not None:
        _same_path(record.get("path"), expected_path, label)
    if not path.is_file():
        raise FinalizationError(f"{label}: file does not exist: {path}")
    recorded_hash = _require_text(record.get("sha256"), f"{label}.sha256").casefold()
    actual_hash = preparation.sha256_file(path)
    if actual_hash != recorded_hash:
        raise FinalizationError(
            f"{label}: SHA256 mismatch; recorded={recorded_hash}, actual={actual_hash}"
        )
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise FinalizationError(
            f"{label}: frozen SHA256 mismatch; expected={expected_sha256}, "
            f"actual={actual_hash}"
        )
    return path, actual_hash


def _recorded_rows(record: Any, actual: int, label: str) -> None:
    if not isinstance(record, dict):
        raise FinalizationError(f"{label}: missing manifest record")
    recorded = record.get("rows")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        raise FinalizationError(f"{label}: rows must be an integer")
    if recorded != actual:
        raise FinalizationError(
            f"{label}: row-count mismatch; recorded={recorded}, actual={actual}"
        )


def _read_jsonl_allow_empty(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FinalizationError(f"{label}: file does not exist: {path}")
    try:
        return merger.read_jsonl(path, label)
    except merger.MergeError as exc:
        raise FinalizationError(str(exc)) from exc


def _index(
    rows: list[dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        return merger.index_rows(rows, label)
    except merger.MergeError as exc:
        raise FinalizationError(str(exc)) from exc


def _assert_order(actual: Sequence[str], expected: Sequence[str], label: str) -> None:
    if list(actual) == list(expected):
        return
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise FinalizationError(
            f"{label}: task-ID mismatch; missing={sorted(set(expected) - set(actual))[:10]}, "
            f"extra={sorted(set(actual) - set(expected))[:10]}"
        )
    raise FinalizationError(f"{label}: task order differs from prepared queue")


def _validate_identity(
    manifest: dict[str, Any], *, mode: str, setup: str, label: str
) -> None:
    expected = {"mode": mode, "setup": setup, "label": label}
    for key, value in expected.items():
        actual = _require_text(manifest.get(key), f"prepare manifest {key}")
        if actual != value:
            raise FinalizationError(
                f"prepare manifest {key} mismatch; expected={value!r}, actual={actual!r}"
            )


def validate_prepared_evaluation(
    *,
    output_dir: Path,
    mode: str,
    setup: str,
    label: str,
    fresh_judge_path: Path,
) -> dict[str, Any]:
    """Revalidate a complete prepare stage and its fresh judge result."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / preparation.OUTPUT_NAMES["manifest"]
    manifest = read_json(manifest_path, "prepare manifest")
    if manifest.get("schema_version") != preparation.SCHEMA_VERSION:
        raise FinalizationError(
            "prepare manifest schema mismatch; expected "
            f"{preparation.SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    _validate_identity(manifest, mode=mode, setup=setup, label=label)
    if manifest.get("generation_gold_access") is not False:
        raise FinalizationError("prepare manifest does not prove generation gold isolation")
    if manifest.get("evaluation_gold_access") is not True:
        raise FinalizationError("prepare manifest has invalid evaluation stage declaration")

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise FinalizationError("prepare manifest: missing sources")
    source_pins = {
        "benchmark": preparation.FROZEN_BENCHMARK_SHA256,
        "baseline_solver": preparation.FROZEN_BASELINE_SOLVER_SHA256,
        "baseline_judge": preparation.FROZEN_BASELINE_JUDGE_SHA256,
        "image_template": preparation.FROZEN_IMAGE_TEMPLATE_SHA256,
        "solver": None,
    }
    verified_sources: dict[str, Path] = {}
    for name, pin in source_pins.items():
        path, _ = _verify_recorded_file(
            sources.get(name), label=f"source {name}", expected_sha256=pin
        )
        verified_sources[name] = path

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FinalizationError("prepare manifest: missing artifacts")
    prepared_paths: dict[str, Path] = {}
    for name in ("image97_input", "fresh_input", "reusable_judge", "deterministic"):
        expected_path = output_dir / preparation.OUTPUT_NAMES[name]
        path, _ = _verify_recorded_file(
            artifacts.get(name),
            label=f"prepared artifact {name}",
            expected_path=expected_path,
        )
        prepared_paths[name] = path

    benchmark_rows = preparation.read_jsonl(
        verified_sources["benchmark"], "benchmark"
    )
    _recorded_rows(sources["benchmark"], len(benchmark_rows), "source benchmark")
    benchmark, task_order = preparation.index_rows(benchmark_rows, "benchmark")
    del benchmark
    if len(task_order) != preparation.DEFAULT_EXPECTED_ROWS:
        raise FinalizationError(
            f"frozen benchmark changed: expected {preparation.DEFAULT_EXPECTED_ROWS}, "
            f"found {len(task_order)}"
        )
    solver_rows = preparation.read_jsonl(verified_sources["solver"], "solver")
    _recorded_rows(sources["solver"], len(solver_rows), "source solver")
    solver, solver_order = preparation.index_rows(solver_rows, "solver")
    preparation._assert_exact_ids(solver_order, task_order, "solver")
    preparation._validate_online_solver(solver)

    baseline_solver_rows = preparation.read_jsonl(
        verified_sources["baseline_solver"], "baseline solver"
    )
    _recorded_rows(
        sources["baseline_solver"],
        len(baseline_solver_rows),
        "source baseline solver",
    )
    baseline_solver, baseline_solver_order = preparation.index_rows(
        baseline_solver_rows, "baseline solver"
    )
    preparation._assert_same_ids(
        baseline_solver_order, task_order, "baseline solver"
    )
    try:
        scorer._validate_no_gold(baseline_solver)
    except scorer.ScoringError as exc:
        raise FinalizationError(str(exc)) from exc

    baseline_rows = preparation.read_jsonl(
        verified_sources["baseline_judge"], "baseline judge"
    )
    _recorded_rows(
        sources["baseline_judge"], len(baseline_rows), "source baseline judge"
    )
    baseline, baseline_order = preparation.index_rows(baseline_rows, "baseline judge")
    preparation._assert_same_ids(baseline_order, task_order, "baseline judge")
    image_ids = [
        task_id
        for task_id in task_order
        if scorer._baseline_source(baseline[task_id], task_id) == "image_judge"
    ]
    if len(image_ids) != preparation.DEFAULT_EXPECTED_IMAGE_JUDGE:
        raise FinalizationError(
            f"frozen image partition changed: expected 97, found {len(image_ids)}"
        )

    source_template_rows = preparation.read_jsonl(
        verified_sources["image_template"], "source image template"
    )
    _recorded_rows(
        sources["image_template"],
        len(source_template_rows),
        "source image template",
    )
    _, source_template_order = preparation.index_rows(
        source_template_rows, "source image template"
    )
    _assert_order(source_template_order, image_ids, "source image template")

    image_rows = preparation.read_jsonl(
        prepared_paths["image97_input"], "prepared image97 input"
    )
    _recorded_rows(
        artifacts["image97_input"], len(image_rows), "prepared image97 input"
    )
    image_by_id, image_order = preparation.index_rows(
        image_rows, "prepared image97 input"
    )
    _assert_order(image_order, image_ids, "prepared image97 input")
    for task_id in image_order:
        if image_by_id[task_id].get("setup") != setup:
            raise FinalizationError(
                f"prepared image97 input {task_id}: setup differs from {setup!r}"
            )

    fresh_queue_rows = _read_jsonl_allow_empty(
        prepared_paths["fresh_input"], "prepared fresh queue"
    )
    _recorded_rows(
        artifacts["fresh_input"], len(fresh_queue_rows), "prepared fresh queue"
    )
    fresh_queue, fresh_order = _index(fresh_queue_rows, "prepared fresh queue")
    reusable_rows = _read_jsonl_allow_empty(
        prepared_paths["reusable_judge"], "prepared reusable judge"
    )
    _recorded_rows(
        artifacts["reusable_judge"],
        len(reusable_rows),
        "prepared reusable judge",
    )
    reusable, reusable_order = _index(reusable_rows, "prepared reusable judge")

    reuse_policy = manifest.get("reuse_policy")
    if not isinstance(reuse_policy, dict):
        raise FinalizationError("prepare manifest: missing reuse_policy")
    changed_ids = [str(value) for value in reuse_policy.get("changed_task_ids") or []]
    unchanged_ids = [str(value) for value in reuse_policy.get("unchanged_task_ids") or []]
    if reuse_policy.get("changed_rows") != len(changed_ids):
        raise FinalizationError("prepare manifest changed_rows does not match task IDs")
    if reuse_policy.get("unchanged_rows") != len(unchanged_ids):
        raise FinalizationError("prepare manifest unchanged_rows does not match task IDs")
    _assert_order(fresh_order, changed_ids, "prepared fresh queue")
    _assert_order(reusable_order, unchanged_ids, "prepared reusable judge")
    if [task_id for task_id in image_order if task_id in set(changed_ids)] != changed_ids:
        raise FinalizationError("changed task IDs do not preserve image97 order")
    if [task_id for task_id in image_order if task_id in set(unchanged_ids)] != unchanged_ids:
        raise FinalizationError("unchanged task IDs do not preserve image97 order")
    if set(changed_ids).intersection(unchanged_ids):
        raise FinalizationError("prepared fresh/reusable partitions overlap")
    if set(changed_ids).union(unchanged_ids) != set(image_order):
        raise FinalizationError("prepared fresh/reusable partitions are incomplete")
    for task_id, row in fresh_queue.items():
        if row != image_by_id[task_id]:
            raise FinalizationError(
                f"prepared fresh queue {task_id}: row differs from full image97 input"
            )
    for task_id, row in reusable.items():
        try:
            merger.validate_judge_row(
                row,
                label="prepared reusable judge",
                task_id=task_id,
                expected_prompt_version="judge-v2",
            )
        except merger.MergeError as exc:
            raise FinalizationError(str(exc)) from exc

    deterministic = read_json(
        prepared_paths["deterministic"], "prepared deterministic audit"
    )
    deterministic_n = deterministic.get("n")
    if deterministic_n != preparation.DEFAULT_EXPECTED_DETERMINISTIC:
        raise FinalizationError(
            "prepared deterministic audit row count changed; "
            f"expected={preparation.DEFAULT_EXPECTED_DETERMINISTIC}, "
            f"actual={deterministic_n!r}"
        )
    _recorded_rows(
        artifacts["deterministic"],
        deterministic_n,
        "prepared deterministic audit",
    )
    manifest_deterministic = manifest.get("deterministic")
    if not isinstance(manifest_deterministic, dict) or (
        manifest_deterministic.get("n") != deterministic_n
        or manifest_deterministic.get("correct") != deterministic.get("correct")
    ):
        raise FinalizationError(
            "prepare manifest deterministic summary differs from hashed audit"
        )

    fresh_result_rows = _read_jsonl_allow_empty(
        fresh_judge_path, "fresh judge-v2 result"
    )
    fresh_results, fresh_result_order = _index(
        fresh_result_rows, "fresh judge-v2 result"
    )
    _assert_order(fresh_result_order, fresh_order, "fresh judge-v2 result")
    adapter_hashes = validate_frozen_judge_adapter()
    linked_requests: dict[str, dict[str, str]] = {}
    for row in fresh_result_rows:
        task_id = str(row["task_id"])
        try:
            merger.validate_judge_row(
                row,
                label="fresh judge-v2 result",
                task_id=task_id,
                expected_prompt_version="judge-v2",
            )
        except merger.MergeError as exc:
            raise FinalizationError(str(exc)) from exc
        linked_requests[task_id] = validate_fresh_judge_request_linkage(
            result=fresh_results[task_id],
            queue_row=fresh_queue[task_id],
        )

    return {
        "prepare_manifest": manifest,
        "prepare_manifest_path": manifest_path,
        "sources": verified_sources,
        "prepared": prepared_paths,
        "task_order": task_order,
        "image_order": image_order,
        "fresh_order": fresh_order,
        "reusable_order": reusable_order,
        "fresh_judge_path": fresh_judge_path.resolve(),
        "frozen_judge_adapter_sha256": adapter_hashes,
        "fresh_request_linkage": linked_requests,
    }


def _archive_existing(
    paths: dict[str, Path], *, output_dir: Path, overwrite_final: bool
) -> Path | None:
    existing = {name: path for name, path in paths.items() if path.exists()}
    if not existing:
        return None
    if not overwrite_final:
        raise FinalizationError(
            "final artifacts already exist; pass --overwrite-final to archive and "
            "replace them: " + ", ".join(str(path) for path in existing.values())
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir / f"previous-final-{stamp}-{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=False, exist_ok=False)
    moved: list[Path] = []
    try:
        for path in existing.values():
            os.replace(path, archive / path.name)
            moved.append(path)
    except Exception:
        for path in reversed(moved):
            archived = archive / path.name
            if archived.exists():
                os.replace(archived, path)
        archive.rmdir()
        raise
    return archive


def _restore_after_failure(
    *, final_paths: dict[str, Path], archive: Path | None, output_dir: Path
) -> Path | None:
    generated = [path for path in final_paths.values() if path.exists()]
    failed_dir: Path | None = None
    if generated:
        failed_dir = output_dir / f"failed-final-{uuid.uuid4().hex[:8]}"
        failed_dir.mkdir(parents=False, exist_ok=False)
        for path in generated:
            os.replace(path, failed_dir / path.name)
    if archive is not None:
        for old in archive.iterdir():
            os.replace(old, output_dir / old.name)
        archive.rmdir()
    return failed_dir


def _finalize_stage_locked(
    *,
    output_dir: Path,
    mode: str,
    setup: str,
    label: str,
    fresh_judge_path: Path,
    overwrite_final: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    validated = validate_prepared_evaluation(
        output_dir=output_dir,
        mode=mode,
        setup=setup,
        label=label,
        fresh_judge_path=fresh_judge_path,
    )
    final_paths = {
        name: output_dir / filename for name, filename in FINAL_OUTPUT_NAMES.items()
    }
    archive = _archive_existing(
        final_paths, output_dir=output_dir, overwrite_final=overwrite_final
    )
    try:
        merge_report = merger.build_from_paths(
            template_path=validated["prepared"]["image97_input"],
            reusable_path=validated["prepared"]["reusable_judge"],
            fresh_path=validated["fresh_judge_path"],
            out_jsonl=final_paths["matched_judge"],
            out_manifest=final_paths["matched_manifest"],
            out_sha256=final_paths["matched_checksums"],
            expected_prompt_version="judge-v2",
        )
        score_report = scorer.build_report(
            benchmark_path=validated["sources"]["benchmark"],
            solver_results_path=validated["sources"]["solver"],
            image_judge_path=final_paths["matched_judge"],
            baseline_judge_path=validated["sources"]["baseline_judge"],
            expected_rows=preparation.DEFAULT_EXPECTED_ROWS,
            expected_deterministic=preparation.DEFAULT_EXPECTED_DETERMINISTIC,
            expected_image_judge=preparation.DEFAULT_EXPECTED_IMAGE_JUDGE,
            expected_benchmark_sha256=preparation.FROZEN_BENCHMARK_SHA256,
            expected_baseline_judge_sha256=preparation.FROZEN_BASELINE_JUDGE_SHA256,
            label=label,
        )
        score_hashes = scorer.write_reports(
            score_report,
            out_json=final_paths["score_json"],
            out_md=final_paths["score_markdown"],
            out_sha256=final_paths["score_checksums"],
        )
        final_manifest = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "setup": setup,
            "label": label,
            "prepare_manifest": {
                "path": str(validated["prepare_manifest_path"]),
                "sha256": preparation.sha256_file(
                    validated["prepare_manifest_path"]
                ),
            },
            "fresh_judge_v2_result": {
                "path": str(validated["fresh_judge_path"]),
                "sha256": preparation.sha256_file(validated["fresh_judge_path"]),
                "rows": len(validated["fresh_order"]),
            },
            "frozen_judge_v2": {
                "prompt_version": FROZEN_JUDGE_PROMPT_VERSION,
                "adapter_sha256": validated["frozen_judge_adapter_sha256"],
                "backend_config": FROZEN_JUDGE_BACKEND_CONFIG,
                "backend_config_sha256": FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
                "request_linkage": validated["fresh_request_linkage"],
            },
            "matched_judge": merge_report,
            "score": {
                "overall": score_report["overall"],
                "output_hashes": score_hashes,
                "paths": {
                    "json": str(final_paths["score_json"]),
                    "markdown": str(final_paths["score_markdown"]),
                    "sha256": str(final_paths["score_checksums"]),
                },
            },
            "previous_outputs_archive": str(archive) if archive else None,
        }
        manifest_data = preparation._json_bytes(final_manifest)
        manifest_hash = preparation.sha256_bytes(manifest_data)
        checksum_data = (
            f"{manifest_hash}  {final_paths['finalization_manifest'].name}\n"
        ).encode("utf-8")
        preparation._atomic_write_many(
            [
                (final_paths["finalization_manifest"], manifest_data),
                (final_paths["finalization_checksums"], checksum_data),
            ]
        )
    except Exception:
        _restore_after_failure(
            final_paths=final_paths, archive=archive, output_dir=output_dir
        )
        raise

    overall = score_report["overall"]
    return {
        "mode": mode,
        "label": label,
        "score": f"{overall['new_correct']}/{overall['n']}",
        "accuracy": overall["new_accuracy"],
        "fresh_judge_rows": len(validated["fresh_order"]),
        "reused_judge_rows": len(validated["reusable_order"]),
        "matched_judge": str(final_paths["matched_judge"]),
        "score_json": str(final_paths["score_json"]),
        "finalization_manifest": str(final_paths["finalization_manifest"]),
        "previous_outputs_archive": str(archive) if archive else None,
    }


def finalize_stage(
    *,
    output_dir: Path,
    mode: str,
    setup: str,
    label: str,
    fresh_judge_path: Path,
    overwrite_final: bool = False,
) -> dict[str, Any]:
    """Finalize exactly once per output directory at the interprocess boundary."""

    resolved_output_dir = output_dir.resolve()
    with ExclusiveOutputDirLock(
        resolved_output_dir,
        mode=mode,
        label=label,
    ):
        return _finalize_stage_locked(
            output_dir=resolved_output_dir,
            mode=mode,
            setup=setup,
            label=label,
            fresh_judge_path=fresh_judge_path,
            overwrite_final=overwrite_final,
        )


def prepare_stage(
    *,
    benchmark_path: Path,
    solver_path: Path,
    baseline_solver_path: Path,
    baseline_judge_path: Path,
    image_template_path: Path,
    output_dir: Path,
    mode: str,
    setup: str,
    label: str,
    expected_rows: int = preparation.DEFAULT_EXPECTED_ROWS,
    expected_deterministic: int = preparation.DEFAULT_EXPECTED_DETERMINISTIC,
    expected_image_judge: int = preparation.DEFAULT_EXPECTED_IMAGE_JUDGE,
    expected_benchmark_sha256: str | None = preparation.FROZEN_BENCHMARK_SHA256,
    expected_baseline_solver_sha256: str | None = preparation.FROZEN_BASELINE_SOLVER_SHA256,
    expected_baseline_judge_sha256: str | None = preparation.FROZEN_BASELINE_JUDGE_SHA256,
    expected_image_template_sha256: str | None = preparation.FROZEN_IMAGE_TEMPLATE_SHA256,
) -> dict[str, Any]:
    return preparation.prepare_from_paths(
        benchmark_path=benchmark_path,
        solver_path=solver_path,
        baseline_solver_path=baseline_solver_path,
        baseline_judge_path=baseline_judge_path,
        image_template_path=image_template_path,
        output_dir=output_dir,
        setup=setup,
        mode=mode,
        label=label,
        expected_rows=expected_rows,
        expected_deterministic=expected_deterministic,
        expected_image_judge=expected_image_judge,
        expected_benchmark_sha256=expected_benchmark_sha256,
        expected_baseline_solver_sha256=expected_baseline_solver_sha256,
        expected_baseline_judge_sha256=expected_baseline_judge_sha256,
        expected_image_template_sha256=expected_image_template_sha256,
    )


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="solver -> judge queues")
    _add_identity(prepare_parser)
    prepare_parser.add_argument("--benchmark", type=Path, required=True)
    prepare_parser.add_argument("--solver-results", type=Path, required=True)
    prepare_parser.add_argument("--baseline-solver", type=Path, required=True)
    prepare_parser.add_argument("--baseline-judge", type=Path, required=True)
    prepare_parser.add_argument("--image-template", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-benchmark-sha256", default=preparation.FROZEN_BENCHMARK_SHA256
    )
    prepare_parser.add_argument(
        "--expected-baseline-solver-sha256",
        default=preparation.FROZEN_BASELINE_SOLVER_SHA256,
    )
    prepare_parser.add_argument(
        "--expected-baseline-judge-sha256",
        default=preparation.FROZEN_BASELINE_JUDGE_SHA256,
    )
    prepare_parser.add_argument(
        "--expected-image-template-sha256",
        default=preparation.FROZEN_IMAGE_TEMPLATE_SHA256,
    )

    finalize_parser = subparsers.add_parser(
        "finalize", help="fresh judge-v2 result -> matched judge + score"
    )
    _add_identity(finalize_parser)
    finalize_parser.add_argument(
        "--fresh-judge-v2-result", type=Path, required=True
    )
    finalize_parser.add_argument("--overwrite-final", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stage == "prepare":
            report = prepare_stage(
                benchmark_path=args.benchmark,
                solver_path=args.solver_results,
                baseline_solver_path=args.baseline_solver,
                baseline_judge_path=args.baseline_judge,
                image_template_path=args.image_template,
                output_dir=args.output_dir,
                mode=args.mode,
                setup=args.setup,
                label=args.label,
                expected_benchmark_sha256=args.expected_benchmark_sha256,
                expected_baseline_solver_sha256=args.expected_baseline_solver_sha256,
                expected_baseline_judge_sha256=args.expected_baseline_judge_sha256,
                expected_image_template_sha256=args.expected_image_template_sha256,
            )
        else:
            report = finalize_stage(
                output_dir=args.output_dir,
                mode=args.mode,
                setup=args.setup,
                label=args.label,
                fresh_judge_path=args.fresh_judge_v2_result,
                overwrite_final=args.overwrite_final,
            )
    except (OSError, FinalizationError, preparation.PreparationError, scorer.ScoringError, merger.MergeError) as exc:
        print(f"ONLINE EVALUATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

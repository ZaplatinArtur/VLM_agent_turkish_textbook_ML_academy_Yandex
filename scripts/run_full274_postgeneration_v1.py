#!/usr/bin/env python3
"""Universal, fail-closed post-generation runner for a completed full274 solver.

This wrapper deliberately does not generate solver answers and never starts a
GPU service.  It composes the existing frozen protocol without changing any
scoring semantics:

1. ``prepare`` delegates to ``finalize_maxim_online_evaluation_v1.prepare``;
2. ``judge`` runs the repository's exact pinned judge-v2 command;
3. ``finalize`` delegates to the existing merge/scorer finalizer;
4. ``all`` performs those three stages in order.

Every stage accepts the same four primary arguments: ``--solver`` (alias
``--solver-results``),
``--setup``, ``--output-dir``, and ``--base-url``.  ``--mode`` and ``--label``
default to the setup string.  The judge model, prompt, seed, decoding settings,
image mode, adapter source, benchmark, baseline, and image partition remain
frozen.  Consequently the base URL must resolve to the endpoint already pinned
by the finalizer (``http://127.0.0.1:18005/v1``); changing it would create a
different judge lineage and is rejected before any request is sent.

Typical runbook::

    python scripts/run_full274_postgeneration_v1.py plan \
      --solver-results outputs/active_vision.jsonl \
      --setup maxim_active_vision_v1 \
      --output-dir reports/active_vision/evaluation \
      --base-url http://127.0.0.1:18005/v1

    # Either run each auditable stage separately ...
    python scripts/run_full274_postgeneration_v1.py prepare ...same arguments...
    python scripts/run_full274_postgeneration_v1.py judge   ...same arguments...
    python scripts/run_full274_postgeneration_v1.py finalize ...same arguments...

    # ... or run the exact same stages sequentially.
    python scripts/run_full274_postgeneration_v1.py all ...same arguments...

The output directory receives the existing preparation/finalization manifests
plus ``postgeneration_orchestration_v1.json`` and its checksum.  Existing judge
output is never replaced implicitly; ``--resume-judge`` reconstructs the whole
queue into a temporary sibling and replaces the old result only after the
existing frozen finalizer validates every row and request binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for import_root in (SCRIPTS_DIR, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import finalize_maxim_online_evaluation_v1 as finalizer  # noqa: E402
import prepare_maxim_online_evaluation_v1 as preparation  # noqa: E402


SCHEMA_VERSION = "full274-postgeneration-orchestration-v1"
STATE_NAME = "postgeneration_orchestration_v1.json"
STATE_CHECKSUM_NAME = "postgeneration_orchestration_v1.sha256"
LOCK_NAME = ".run_full274_postgeneration_v1.lock"
FRESH_RESULT_NAME = "fresh_judge_v2_result.jsonl"

DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "artifacts"
    / "baselines"
    / "basic_page_rag_v1"
    / "validation_274.jsonl"
)
DEFAULT_BASELINE_SOLVER = (
    REPO_ROOT
    / "artifacts"
    / "baselines"
    / "basic_page_rag_v1"
    / "agent_rag_274.jsonl"
)
DEFAULT_BASELINE_JUDGE = (
    REPO_ROOT
    / "artifacts"
    / "baselines"
    / "basic_page_rag_v1"
    / "agent_rag_judge.jsonl"
)
DEFAULT_IMAGE_TEMPLATE = (
    REPO_ROOT
    / "reports"
    / "maxim_ideas_full274_20260731"
    / "parallel8_reasoning_first_v2"
    / "image97_input.jsonl"
)


class PostGenerationError(ValueError):
    """Raised when the frozen post-generation protocol cannot proceed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise PostGenerationError("--base-url must be non-empty")
    endpoint = str(finalizer.FROZEN_JUDGE_BACKEND_CONFIG["endpoint"])
    suffix = "/chat/completions"
    if not endpoint.endswith(suffix):
        raise PostGenerationError("frozen judge endpoint has an invalid shape")
    frozen_base_url = endpoint[: -len(suffix)].rstrip("/")
    if normalized != frozen_base_url:
        raise PostGenerationError(
            "--base-url differs from the frozen matched judge-v2 lineage; "
            f"expected={frozen_base_url!r}, actual={normalized!r}"
        )
    return normalized


def _non_empty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise PostGenerationError(f"{label} must be non-empty")
    return normalized


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PostGenerationError(f"required file does not exist: {resolved}")
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _jsonl_count(path: Path, *, allow_empty: bool = False) -> int:
    if not path.is_file():
        raise PostGenerationError(f"JSONL file does not exist: {path}")
    count = 0
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PostGenerationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PostGenerationError(
                    f"{path}:{line_number}: row is not an object"
                )
            count += 1
    if not allow_empty and count == 0:
        raise PostGenerationError(f"JSONL file is empty: {path}")
    return count


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    solver_results: Path
    setup: str
    output_dir: Path
    base_url: str
    mode: str
    label: str
    benchmark: Path = DEFAULT_BENCHMARK
    baseline_solver: Path = DEFAULT_BASELINE_SOLVER
    baseline_judge: Path = DEFAULT_BASELINE_JUDGE
    image_template: Path = DEFAULT_IMAGE_TEMPLATE

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        setup = _non_empty(args.setup, "--setup")
        return cls(
            solver_results=Path(args.solver_results).resolve(),
            setup=setup,
            output_dir=Path(args.output_dir).resolve(),
            base_url=_normalize_base_url(args.base_url),
            mode=_non_empty(args.mode or setup, "--mode"),
            label=_non_empty(args.label or setup, "--label"),
            benchmark=Path(args.benchmark).resolve(),
            baseline_solver=Path(args.baseline_solver).resolve(),
            baseline_judge=Path(args.baseline_judge).resolve(),
            image_template=Path(args.image_template).resolve(),
        )


class ExclusiveLock:
    def __init__(self, output_dir: Path, stage: str) -> None:
        self.output_dir = output_dir
        self.path = output_dir / LOCK_NAME
        self.stage = stage
        self.token = uuid.uuid4().hex
        self.fd: int | None = None

    def __enter__(self) -> "ExclusiveLock":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema_version": "full274-postgeneration-lock-v1",
                "token": self.token,
                "pid": os.getpid(),
                "stage": self.stage,
                "created_at": _utc_now(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        try:
            self.fd = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise PostGenerationError(
                f"post-generation output directory is locked: {self.path}"
            ) from exc
        try:
            os.write(self.fd, payload)
            os.fsync(self.fd)
        except Exception:
            os.close(self.fd)
            self.fd = None
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = {}
        if owner.get("token") == self.token:
            self.path.unlink(missing_ok=True)
        elif exc_type is None:
            raise PostGenerationError(
                f"post-generation lock ownership changed: {self.path}"
            )
        return False


def _delegate_records() -> dict[str, dict[str, Any]]:
    return {
        "orchestrator": _file_record(Path(__file__)),
        "preparation": _file_record(Path(preparation.__file__)),
        "finalizer": _file_record(Path(finalizer.__file__)),
        "judge_cli": _file_record(SRC_DIR / "vlm_judge" / "cli.py"),
    }


def _source_records(config: PipelineConfig) -> dict[str, dict[str, Any]]:
    return {
        "solver": _file_record(
            config.solver_results,
            rows=_jsonl_count(config.solver_results),
        ),
        "benchmark": _file_record(
            config.benchmark,
            rows=_jsonl_count(config.benchmark),
        ),
        "baseline_solver": _file_record(
            config.baseline_solver,
            rows=_jsonl_count(config.baseline_solver),
        ),
        "baseline_judge": _file_record(
            config.baseline_judge,
            rows=_jsonl_count(config.baseline_judge),
        ),
        "image_template": _file_record(
            config.image_template,
            rows=_jsonl_count(config.image_template),
        ),
    }


def _identity(config: PipelineConfig) -> dict[str, str]:
    return {
        "mode": config.mode,
        "setup": config.setup,
        "label": config.label,
    }


def _judge_record(config: PipelineConfig) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "model": finalizer.FROZEN_JUDGE_MODEL,
        "prompt_version": finalizer.FROZEN_JUDGE_PROMPT_VERSION,
        "backend_config": dict(finalizer.FROZEN_JUDGE_BACKEND_CONFIG),
        "backend_config_sha256": finalizer.FROZEN_JUDGE_BACKEND_CONFIG_SHA256,
    }


def _state_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / STATE_NAME, output_dir / STATE_CHECKSUM_NAME


def _write_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    state_path, checksum_path = _state_paths(output_dir)
    data = (
        json.dumps(
            dict(state),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    checksum_data = f"{digest}  {STATE_NAME}\n".encode("ascii")
    token = uuid.uuid4().hex
    state_tmp = state_path.with_name(f".{state_path.name}.{token}.tmp")
    checksum_tmp = checksum_path.with_name(
        f".{checksum_path.name}.{token}.tmp"
    )
    try:
        state_tmp.write_bytes(data)
        checksum_tmp.write_bytes(checksum_data)
        os.replace(state_tmp, state_path)
        os.replace(checksum_tmp, checksum_path)
    finally:
        state_tmp.unlink(missing_ok=True)
        checksum_tmp.unlink(missing_ok=True)


def _load_state(config: PipelineConfig) -> dict[str, Any]:
    state_path, checksum_path = _state_paths(config.output_dir)
    if not state_path.is_file() or not checksum_path.is_file():
        raise PostGenerationError(
            "post-generation state is missing; run the prepare stage first"
        )
    expected_line = checksum_path.read_text(encoding="ascii").strip()
    expected = f"{sha256_file(state_path)}  {STATE_NAME}"
    if expected_line != expected:
        raise PostGenerationError("post-generation state checksum mismatch")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostGenerationError(
            f"post-generation state is invalid JSON: {exc}"
        ) from exc
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise PostGenerationError("post-generation state schema mismatch")
    if state.get("identity") != _identity(config):
        raise PostGenerationError("post-generation identity differs from prepare")
    if state.get("judge") != _judge_record(config):
        raise PostGenerationError("post-generation judge binding differs from prepare")

    current_delegates = _delegate_records()
    if state.get("delegates") != current_delegates:
        raise PostGenerationError(
            "post-generation orchestrator or frozen delegate source changed "
            "after prepare"
        )

    current_sources = _source_records(config)
    recorded_sources = state.get("sources")
    if not isinstance(recorded_sources, dict):
        raise PostGenerationError("post-generation source provenance is missing")
    for name, record in current_sources.items():
        if recorded_sources.get(name) != record:
            raise PostGenerationError(
                f"post-generation source changed after prepare: {name}"
            )
    prepare_stage = (state.get("stages") or {}).get("prepare")
    if not isinstance(prepare_stage, dict) or prepare_stage.get("status") != "complete":
        raise PostGenerationError("prepare stage is not complete")
    for record in (prepare_stage.get("artifacts") or {}).values():
        if not isinstance(record, dict):
            raise PostGenerationError("invalid recorded prepare artifact")
        path = Path(str(record.get("path") or ""))
        if _file_record(path) != record:
            raise PostGenerationError(
                f"prepared artifact changed after prepare: {path}"
            )
    return state


def _append_event(
    state: dict[str, Any], stage: str, details: Mapping[str, Any]
) -> None:
    state.setdefault("events", []).append(
        {
            "stage": stage,
            "completed_at": _utc_now(),
            **dict(details),
        }
    )
    state["updated_at"] = _utc_now()


def _prepared_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    names = {
        **preparation.OUTPUT_NAMES,
    }
    return {
        name: _file_record(output_dir / filename)
        for name, filename in names.items()
    }


def prepare_stage(config: PipelineConfig) -> dict[str, Any]:
    state_path, checksum_path = _state_paths(config.output_dir)
    if state_path.exists() or checksum_path.exists():
        raise PostGenerationError(
            "post-generation state already exists; refusing implicit re-prepare"
        )
    report = finalizer.prepare_stage(
        benchmark_path=config.benchmark,
        solver_path=config.solver_results,
        baseline_solver_path=config.baseline_solver,
        baseline_judge_path=config.baseline_judge,
        image_template_path=config.image_template,
        output_dir=config.output_dir,
        mode=config.mode,
        setup=config.setup,
        label=config.label,
        expected_rows=preparation.DEFAULT_EXPECTED_ROWS,
        expected_deterministic=preparation.DEFAULT_EXPECTED_DETERMINISTIC,
        expected_image_judge=preparation.DEFAULT_EXPECTED_IMAGE_JUDGE,
        expected_benchmark_sha256=preparation.FROZEN_BENCHMARK_SHA256,
        expected_baseline_solver_sha256=preparation.FROZEN_BASELINE_SOLVER_SHA256,
        expected_baseline_judge_sha256=preparation.FROZEN_BASELINE_JUDGE_SHA256,
        expected_image_template_sha256=preparation.FROZEN_IMAGE_TEMPLATE_SHA256,
    )
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "identity": _identity(config),
        "judge": _judge_record(config),
        "sources": _source_records(config),
        "delegates": _delegate_records(),
        "stages": {
            "prepare": {
                "status": "complete",
                "report": report,
                "artifacts": _prepared_artifacts(config.output_dir),
            }
        },
        "events": [],
    }
    _append_event(state, "prepare", {"report": report})
    _write_state(config.output_dir, state)
    return {
        "stage": "prepare",
        "fresh_judge_rows": report["fresh_judge_rows"],
        "manifest": report["manifest"],
        "orchestration": str((config.output_dir / STATE_NAME).resolve()),
    }


def _judge_command(
    config: PipelineConfig,
    *,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    cache_dir = config.output_dir / "judge_cache"
    return [
        sys.executable,
        "-m",
        "vlm_judge.cli",
        "run-judge",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--base-url",
        config.base_url,
        "--model",
        finalizer.FROZEN_JUDGE_MODEL,
        "--timeout",
        "120",
        "--temperature",
        "0",
        "--max-tokens",
        "900",
        "--seed",
        "20260714",
        "--image-mode",
        "data_url",
        "--image-cache-dir",
        str(cache_dir / "images"),
        "--disable-thinking",
        "--cache-dir",
        str(cache_dir / "responses"),
        "--prompt-version",
        "judge-v2",
        "--max-attempts",
        "2",
        "--workers",
        "1",
        "--retry-delay",
        "1",
    ]


def judge_stage(
    config: PipelineConfig,
    *,
    resume_judge: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    state = _load_state(config)
    finalizer.validate_frozen_judge_adapter()
    input_path = config.output_dir / preparation.OUTPUT_NAMES["fresh_input"]
    expected_rows = _jsonl_count(input_path, allow_empty=True)
    output_path = config.output_dir / FRESH_RESULT_NAME
    if output_path.exists() and not resume_judge:
        raise PostGenerationError(
            "fresh judge output exists; pass --resume-judge to reconstruct it "
            "from the complete queue and frozen cache"
        )
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    command = _judge_command(
        config,
        input_path=input_path,
        output_path=temporary,
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    try:
        if expected_rows == 0:
            temporary.write_bytes(b"")
        else:
            runner(command, cwd=REPO_ROOT, env=env, check=True)
        actual_rows = _jsonl_count(temporary, allow_empty=True)
        if actual_rows != expected_rows:
            raise PostGenerationError(
                "fresh judge row count differs from prepared queue; "
                f"expected={expected_rows}, actual={actual_rows}"
            )
        # This is the existing frozen validator.  It checks exact queue order,
        # request IDs, backend config, cache keys, served model, and verdicts.
        finalizer.validate_prepared_evaluation(
            output_dir=config.output_dir,
            mode=config.mode,
            setup=config.setup,
            label=config.label,
            fresh_judge_path=temporary,
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    state.setdefault("stages", {})["judge"] = {
        "status": "complete",
        "rows": expected_rows,
        "output": _file_record(output_path),
        "command": _judge_command(
            config,
            input_path=input_path,
            output_path=output_path,
        ),
        "resumed": bool(resume_judge),
    }
    _append_event(
        state,
        "judge",
        {
            "rows": expected_rows,
            "output_sha256": sha256_file(output_path),
            "resumed": bool(resume_judge),
        },
    )
    _write_state(config.output_dir, state)
    return {
        "stage": "judge",
        "rows": expected_rows,
        "output": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
    }


def finalize_stage(
    config: PipelineConfig,
    *,
    overwrite_final: bool = False,
) -> dict[str, Any]:
    state = _load_state(config)
    judge_stage_record = (state.get("stages") or {}).get("judge")
    if (
        not isinstance(judge_stage_record, dict)
        or judge_stage_record.get("status") != "complete"
    ):
        raise PostGenerationError("judge stage is not complete")
    fresh_result = config.output_dir / FRESH_RESULT_NAME
    if _file_record(fresh_result) != judge_stage_record.get("output"):
        raise PostGenerationError("fresh judge output changed after validation")
    report = finalizer.finalize_stage(
        output_dir=config.output_dir,
        mode=config.mode,
        setup=config.setup,
        label=config.label,
        fresh_judge_path=fresh_result,
        overwrite_final=overwrite_final,
    )
    final_names = finalizer.FINAL_OUTPUT_NAMES
    final_artifacts = {
        name: _file_record(config.output_dir / filename)
        for name, filename in final_names.items()
    }
    state.setdefault("stages", {})["finalize"] = {
        "status": "complete",
        "report": report,
        "artifacts": final_artifacts,
        "overwrite_final": bool(overwrite_final),
    }
    _append_event(state, "finalize", {"report": report})
    _write_state(config.output_dir, state)
    return {
        "stage": "finalize",
        **report,
        "orchestration": str((config.output_dir / STATE_NAME).resolve()),
    }


def plan(config: PipelineConfig) -> dict[str, Any]:
    sources = _source_records(config)
    queue = config.output_dir / preparation.OUTPUT_NAMES["fresh_input"]
    result = config.output_dir / FRESH_RESULT_NAME
    return {
        "schema_version": SCHEMA_VERSION,
        "network_or_gpu_actions_performed": False,
        "identity": _identity(config),
        "judge": _judge_record(config),
        "sources": sources,
        "output_dir": str(config.output_dir),
        "stages": [
            {
                "stage": "prepare",
                "delegate": str(Path(finalizer.__file__).resolve()),
                "fresh_queue": str(queue),
            },
            {
                "stage": "judge",
                "command": _judge_command(
                    config,
                    input_path=queue,
                    output_path=result,
                ),
            },
            {
                "stage": "finalize",
                "delegate": str(Path(finalizer.__file__).resolve()),
                "fresh_result": str(result),
            },
        ],
        "notes": [
            "solver generation must already be complete and explicitly gold-isolated",
            "prepare may read frozen gold only after generation",
            "judge-v2 model/prompt/seed/backend settings are immutable",
            "the wrapper never starts or manages a GPU service",
            "final score semantics are owned solely by the existing frozen finalizer/scorer",
        ],
    }


def status(config: PipelineConfig) -> dict[str, Any]:
    state = _load_state(config)
    return {
        "state": str((config.output_dir / STATE_NAME).resolve()),
        "state_sha256": sha256_file(config.output_dir / STATE_NAME),
        "identity": state["identity"],
        "stages": state.get("stages") or {},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("plan", "prepare", "judge", "finalize", "all", "status"),
    )
    parser.add_argument(
        "--solver",
        "--solver-results",
        dest="solver_results",
        type=Path,
        required=True,
    )
    parser.add_argument("--setup", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode")
    parser.add_argument("--label")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--baseline-solver", type=Path, default=DEFAULT_BASELINE_SOLVER
    )
    parser.add_argument(
        "--baseline-judge", type=Path, default=DEFAULT_BASELINE_JUDGE
    )
    parser.add_argument(
        "--image-template", type=Path, default=DEFAULT_IMAGE_TEMPLATE
    )
    parser.add_argument(
        "--resume-judge",
        action="store_true",
        help="rebuild the complete fresh queue using the frozen judge cache",
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="delegate recoverable archival/replacement to the existing finalizer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = PipelineConfig.from_args(args)
        if args.stage == "plan":
            report = plan(config)
        elif args.stage == "status":
            report = status(config)
        else:
            with ExclusiveLock(config.output_dir, args.stage):
                if args.stage == "prepare":
                    report = prepare_stage(config)
                elif args.stage == "judge":
                    report = judge_stage(
                        config,
                        resume_judge=args.resume_judge,
                    )
                elif args.stage == "finalize":
                    report = finalize_stage(
                        config,
                        overwrite_final=args.overwrite_final,
                    )
                else:
                    prepare = prepare_stage(config)
                    judge = judge_stage(
                        config,
                        resume_judge=args.resume_judge,
                    )
                    finalized = finalize_stage(
                        config,
                        overwrite_final=args.overwrite_final,
                    )
                    report = {
                        "stage": "all",
                        "prepare": prepare,
                        "judge": judge,
                        "finalize": finalized,
                    }
    except (
        OSError,
        PostGenerationError,
        preparation.PreparationError,
        finalizer.FinalizationError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"POST-GENERATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

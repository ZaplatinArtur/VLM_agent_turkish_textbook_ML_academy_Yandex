#!/usr/bin/env python3
"""Run Qwen3.5-27B directly on all 274 frozen tasks, with no gold access.

This profile reuses the same answer-blind B0 prompt, image binding, structured
schema, transport, canonical writer and source pins as the hard86 experiments.
Unlike the hard86 composite it selects every visible task, making the 27B
artifact an independent full-benchmark candidate for later blind selection.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any, Sequence


BASE_RUNNER = Path(__file__).with_name(
    "run_maxim_native_thinking_math_router_v1.py"
)
SPEC = importlib.util.spec_from_file_location("maxim_stronger_full274_shared", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - deployment failure
    raise RuntimeError(f"cannot load shared frozen runner: {BASE_RUNNER}")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

runner.SCHEMA_VERSION = "maxim-stronger-27b-direct-full274-v1"
runner.PROFILE_PATH = Path(__file__).resolve()
runner.CONDITION = "maxim_stronger_27b_direct_full274_v1"
runner.RULE_ID = "all_frozen_visible_tasks_to_qwen35_27b_direct_v1"
runner.MODEL = "Qwen/Qwen3.5-27B"
runner.MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
runner.TREATMENT_ID = "stronger_model_qwen35_27b_direct_full274_v1"
runner.PRIMARY_ENABLE_THINKING = False
runner.PRIMARY_DECISION = "stronger_27b_direct"
runner.TEMPERATURE = 0.0
runner.MAX_TOKENS = 3072
runner.WRAPUP_MAX_TOKENS = 1024

# The shared engine's selection hook is intentionally replaced in this
# isolated module instance.  The original subject detector remains available
# for the frozen 139-row benchmark audit; only selection becomes all-task.
_subject_is_math = runner.is_math
runner.is_native_hard_case = lambda _task, _router, _page: True
runner.EXPECTED_NATIVE_ROWS = runner.EXPECTED_ROWS

# ``solve_math_task`` uses subject solely as a defensive selector assertion;
# the B0 messages themselves never include subject.  Satisfy that internal
# assertion on a private task copy, without mutating shared globals across
# worker threads, while preserving the original task_id/image/question data.
_shared_solve = runner.solve_math_task


def _solve_selected_task(task: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    admitted = copy.deepcopy(task)
    admitted["subject"] = "Math"
    return _shared_solve(admitted, **kwargs)


runner.solve_math_task = _solve_selected_task

_shared_identity = runner._run_identity


def _full274_identity(**kwargs: Any) -> dict[str, Any]:
    identity = _shared_identity(**kwargs)
    identity = copy.deepcopy(identity)
    identity["routing"] = {
        "selection": "all_frozen_visible_tasks",
        "benchmark_fields_used": ["task_id"],
        "selected_rows": runner.EXPECTED_ROWS,
        "gold_access": False,
        "rule_id": runner.RULE_ID,
    }
    return identity


runner._run_identity = _full274_identity


def main(argv: Sequence[str] | None = None) -> int:
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

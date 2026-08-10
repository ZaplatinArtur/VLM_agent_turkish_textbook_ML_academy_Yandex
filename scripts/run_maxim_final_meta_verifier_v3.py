#!/usr/bin/env python3
"""Run the separate preregistered twelve-candidate meta-verifier v3.

The runner does not accept or load a private identity key, candidate scores,
reference answers, or judge artifacts.  It preserves the frozen confidence,
evidence, format, non-abstention, and exact Router fallback policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - deployment failure
        raise RuntimeError(f"cannot load required module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
preparation = _load(
    "maxim_final_meta_prepare_v3_runtime",
    SCRIPT_DIR / "prepare_maxim_final_meta_verifier_v3.py",
)
implementation = _load(
    "maxim_final_meta_run_v3_base",
    SCRIPT_DIR / "run_maxim_final_meta_verifier_v1.py",
)

implementation.preparation = preparation
implementation.SCHEMA_VERSION = "maxim-final-meta-verifier-runner-v3"
implementation.CONDITION = "maxim_final_gold_blind_meta_verifier_v3"
implementation.PROMPT_VERSION = "original-first-anonymous-12way-verification-v3"

SCHEMA_VERSION = implementation.SCHEMA_VERSION
CONDITION = implementation.CONDITION
PROMPT_VERSION = implementation.PROMPT_VERSION
SYSTEM_PROMPT = implementation.SYSTEM_PROMPT
MetaVerifierError = implementation.MetaVerifierError
verdict_schema = implementation.verdict_schema
build_messages = implementation.build_messages
validate_verdict = implementation.validate_verdict
validate_queue = implementation.validate_queue
run_one = implementation.run_one
apply_frozen_policy = implementation.apply_frozen_policy
compose_solver_row = implementation.compose_solver_row
run_queue = implementation.run_queue
build_parser = implementation.build_parser


def main(argv: Sequence[str] | None = None) -> int:
    implementation.__doc__ = __doc__
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
